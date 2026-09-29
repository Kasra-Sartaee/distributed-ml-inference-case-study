<div align="center">

# Distributed ML Inference Platform

**Batch ResNet-50 inference over ImageNet on a five-node, CPU-only Kubernetes cluster, built end to end:
from provisioning the VMs to a REST API that needs no Kubernetes or Spark knowledge to use.**

[![Kubernetes](https://img.shields.io/badge/Kubernetes-K3s_1.34-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](#architecture)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](#architecture)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--50-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#architecture)
[![Apache Airflow](https://img.shields.io/badge/Apache_Airflow-2.9-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)](#architecture)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](#architecture)
[![MinIO](https://img.shields.io/badge/MinIO-erasure_coded-C72E49?style=flat-square&logo=minio&logoColor=white)](#architecture)
[![FastAPI](https://img.shields.io/badge/FastAPI-gateway-009688?style=flat-square&logo=fastapi&logoColor=white)](#architecture)
<br>
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](#architecture)
[![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?style=flat-square&logo=grafana&logoColor=white)](#architecture)
[![Terraform](https://img.shields.io/badge/Terraform-IaC-844FBA?style=flat-square&logo=terraform&logoColor=white)](#architecture)
[![Ansible](https://img.shields.io/badge/Ansible-config-EE0000?style=flat-square&logo=ansible&logoColor=white)](#architecture)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=flat-square&logo=githubactions&logoColor=white)](#architecture)

| **7.2M** images processed | **25.1 h** unattended run | **79.6 img/s** sustained | **0** failed passes |
|:-:|:-:|:-:|:-:|

[Architecture](#architecture) · [Design decisions](#key-design-decisions) · [Results](#results) · [Findings](#what-the-measurements-actually-showed) · [Fault tolerance](#fault-tolerance) · [Lessons](#lessons)

</div>

---

> [!NOTE]
> This repository is a write-up. The source code is private and is available to
> reviewers on request.

| | |
|---|---|
| **Role** | Sole designer and engineer: infrastructure, pipeline, API, observability, benchmarking |
| **When** | April 2026 |
| **Scale** | 5 VMs, 22 vCPU, 140 GB RAM, no GPUs; 900,000 ImageNet images in 900 shards |
| **Stack** | Terraform, Ansible, K3s, Spark, PyTorch, Airflow, MLflow, MinIO, FastAPI, Prometheus, Grafana, Loki |

## What it does

1. A user submits a job through a REST API: either a path to data already in
   object storage, or a batch of uploaded images.
2. The API triggers a workflow that launches a distributed Spark job on
   Kubernetes.
3. Spark executors read images from distributed object storage in large tar
   shards, batch them through ResNet-50, and write top-5 predictions as Parquet.
4. The run's parameters, statistics and artifacts go to an experiment tracker;
   live throughput and hardware metrics go to dashboards.
5. The user polls for status and downloads the results through the same API.

## Architecture

```mermaid
flowchart LR
    user([User]) -->|REST| api[FastAPI gateway]
    api -->|trigger| airflow[Airflow]
    airflow -->|SparkApplication| operator[Spark Operator]
    operator --> driver[Spark driver]
    driver --> exec[14 executors<br/>one model each]
    exec <-->|tar shards in,<br/>Parquet out| minio[(MinIO<br/>4 nodes, erasure-coded)]
    exec -->|metrics| pgw[Pushgateway]
    driver -->|params, metrics,<br/>artifacts| mlflow[MLflow]
    pgw --> prom[Prometheus] --> grafana[Grafana]
    loki[Loki logs] --> grafana
    api -->|download results| minio
```

**Cluster:** one small control node (2 vCPU, 20 GB) running orchestration and
observability, and four workers (5 vCPU, 30 GB each) running inference and
storage. The control node is deliberately lean: its services mostly wait on
requests and scrape intervals, so the cores go to the workers. Services are
split across five namespaces: compute, storage, MLOps, monitoring and gateway.

| Layer | Technology |
|---|---|
| Infrastructure as code | Terraform for VMs, Ansible for node setup, K3s |
| Storage | MinIO across 4 nodes with erasure coding (tolerates a node loss) |
| Compute | Apache Spark 3.5 on Kubernetes via the Spark Operator; PyTorch on CPU |
| Orchestration | Apache Airflow with the Kubernetes executor |
| Experiment tracking | MLflow (PostgreSQL metadata, MinIO artifacts) |
| Serving | FastAPI: submit, status, and results download |
| Observability | Prometheus, Pushgateway, Grafana, Loki |
| Security | Default-deny network policies per namespace, least-privilege RBAC, resource quotas |
| CI | GitHub Actions: lint, tests, image builds to GHCR |

<details>
<summary><b>How a request flows through the system</b></summary>

<br>

```mermaid
sequenceDiagram
    actor U as User
    participant API as FastAPI gateway
    participant AF as Airflow
    participant OP as Spark Operator
    participant S as Spark (driver + executors)
    participant M as MinIO
    participant T as MLflow

    U->>API: POST /pipeline/submit
    API->>AF: Trigger DAG run (server-generated run ID)
    API-->>U: run ID
    AF->>M: Check the input shards exist
    AF->>OP: Create SparkApplication
    OP->>S: Launch driver; driver requests 14 executors
    S->>M: Read tar shards
    S->>M: Write Parquet predictions
    S->>T: Log parameters, metrics, artifacts
    AF->>OP: Poll until complete, then clean up
    U->>API: GET /pipeline/status/{run_id}
    U->>API: GET /pipeline/results/{run_id}/download
    API->>M: Stream Parquet back to the user
```

</details>

## Key design decisions

**Spark on Kubernetes rather than Ray, Dask or a task queue.** Spark recovers a
lost executor's work by recomputing only the affected partitions (RDD lineage),
which matters for a run lasting more than a day. The Spark Operator lets a job
be declared as a Kubernetes resource, so the workflow engine submits a
manifest instead of managing Spark processes itself.

**Load the model once per executor, not once per image.** Inference runs
per partition, with the model cached in each executor process, so 900,000
images meant roughly one model load per executor rather than one per image.

**Tar shards instead of individual JPEGs.** 900 shards of 1,000 images turn
hundreds of thousands of small object reads into 900 large sequential ones.
Each shard maps to one Spark partition.

**Distributed object storage, after a mid-project rebuild.** The first design
put all 140 GB of data on the control node. That filled its disk and made
storage a single point of failure, so I replaced it with a four-node MinIO
deployment and rewrote ingest as a parallel Kubernetes Job that streams the
dataset straight into storage. No node ever holds the full dataset.

**Single-core executors.** Two-core executors ran two inference tasks in one
container and hit memory-pressure evictions. With one core, each executor runs
one task with one model, so memory is predictable, and 14 of them fit the
namespace quota with headroom.

**Security by default.** Every namespace denies inbound traffic unless a policy
allows it, and each allowed path is listed explicitly: object storage, for
example, accepts traffic only from the handful of workloads that use it. No
credentials live in the repository.

## Results

Each configuration processed the same 900,000 images once.

| Configuration | Executors | Images/sec | Per executor |
|---|---:|---:|---:|
| Baseline | 14 | 78.91 | 5.64 |
| Larger input partitions | 14 | 80.95 | 5.78 |
| + Speculative execution | 14 | 81.22 | 5.80 |
| More memory per executor | 12 | 69.52 | 5.79 |
| **25-hour run (8 passes)** | **14** | **79.60** | **5.69** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/capacity-passes-dark.png">
  <img alt="Column chart of throughput in each of the 8 capacity-run passes. All passes fall between 78.40 images/sec (pass 5, colour jitter) and 81.59 (pass 8, crop)." src="assets/capacity-passes-light.png">
</picture>

Throughput stayed within about 4% across all eight passes. The slowest pass
used the most expensive preprocessing (colour jitter), and the last pass was
the fastest, so there was no sign of memory leaks, storage degradation or
gradual slowdown.

## What the measurements actually showed

> [!IMPORTANT]
> I originally expected storage I/O to be the bottleneck and tuned for it. The
> data says the system was bound by **the number of executors**, and about a
> quarter of the worker cores ran no executor at all.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/per-executor-throughput-dark.png">
  <img alt="Column chart of images per second per executor: 5.64 baseline, 5.78 larger partitions, 5.80 with speculation, 5.79 with more memory and 12 executors." src="assets/per-executor-throughput-light.png">
</picture>

**Throughput scaled with the number of executors and almost nothing else.**
Each executor processed about 5.7 images/second in every configuration. The
"more memory" round dropped from 14 to 12 executors to fit its larger
allocation, and lost almost exactly two executors' worth of throughput
(81.22 × 12⁄14 = 69.6 predicted, 69.52 measured).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/worker-cpu-dark.png">
  <img alt="Column chart of average worker CPU during the baseline: two workers at about 79% and two at about 61%, matching reference lines for 4 of 5 and 3 of 5 cores busy." src="assets/worker-cpu-light.png">
</picture>

**The CPU numbers say the same thing.** Fourteen single-core executors on four
five-core workers land 4, 4, 3 and 3 per node, which is 80% and 60% CPU. That
is what the workers showed, while disk reads averaged under 2 MB/s. The
workload was bound by executor slots, and about 5 of the 20 worker cores ran
no executor.

**The tuning results needed a second look.** The partition-size change
probably did not change partitioning at all: shards are never split and each
is larger than both settings, so the source of its 2.6% gain is unclear.
Speculative execution added 0.3%, inside the roughly 1% spread between
identical passes. The lever that would have mattered is more executors, which
means shrinking each executor's memory footprint and raising the CPU quota.

**The monitoring had a bug that hid this.** Every executor pushed metrics to
the same Pushgateway group, and each push replaced the previous one, so the
live throughput graph substantially under-reported. The headline numbers
were always taken from wall-clock time and the experiment tracker, so they are
unaffected. The fix was a per-executor grouping key.

## Fault tolerance

| Failure | What happens |
|---|---|
| Executor lost | The driver requests a replacement pod and re-runs only the lost partitions |
| Whole Spark job fails | The Spark Operator resubmits it (up to 2 retries) |
| One storage node down | Reads and writes continue; with two down, data stays readable |
| Worker node lost | Kubernetes evicts and reschedules its pods; the job restarts rather than failing |
| Workflow task fails | Airflow retries it |

> [!WARNING]
> The control node is a single point of failure: it hosts the workflow engine,
> tracking server, database and API. That was an accepted trade-off at this
> scale. In production I would separate the control plane and use managed
> metadata stores.

## Lessons

- **Measure the bottleneck before tuning.** I spent optimisation rounds on
  I/O-oriented settings when per-executor throughput and CPU placement already
  pointed at executor count. Normalising results per worker unit would have
  shown this on day one.
- **Check the monitoring as well as the system.** A metrics pipeline can be
  quietly wrong. Cross-checking live metrics against an independent source
  (wall-clock totals) is what exposed the Pushgateway overwrites.
- **Integration failures are the real work.** The 24-hour workflow failed four
  times before it ran cleanly: worker pods not inheriting environment and
  secrets, pods scheduled in the wrong namespace, network-policy labels left
  stale by the storage migration, and a mismatch between two S3 client
  libraries. Each was diagnosed from task logs and cluster state, and I kept a
  script-based fallback so the long run was never blocked on them.
- **Report negative results.** The memory-tuning round made things worse, and
  it turned out to be the clearest evidence of what actually limited the
  system.

## What I would do next

- [ ] Run more, smaller executors (trimming the JVM heap, which inference barely uses) and use the idle cores
- [ ] Repeat each configuration several times to separate real effects from noise
- [ ] Move the control-plane services off a single node
- [ ] Add a small local profile so the platform can be tried without a cluster

---

<div align="center">
<sub>Source code available to reviewers on request · <a href="https://github.com/Kasra-Sartaee">github.com/Kasra-Sartaee</a></sub>
</div>
