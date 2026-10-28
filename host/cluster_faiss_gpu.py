#!/usr/bin/env python3
"""
Streamed Faiss clustering for large vector datasets.

The output layout matches `host/data/kmeans.cc::save_kmeans_results`:
    [uint64 k][uint64 d][uint64 n]
    [k * d float32 centroids]
    [k + 1 uint64 cluster_offsets]
    [cluster_members int32]

Supported inputs:
    - `.fvecs` (deep)
    - `.bvecs` (bigann / sift)
    - SpaceV `.bin`

For dataset size over 100m, the script avoids loading the full dataset into GPU
memory. It trains initial centroids on a sampled subset with Faiss, then
refines and assigns the full dataset in streaming batches.

Usage: python3 host/data/cluster_faiss_gpu.py --data-file /mnt/anns-dataset/bigann/bigann_base.bvecs \
--data-format bvecs --dataset-name sift --dataset-size-label 500M --max-vectors 500000000
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Faiss is required. Install `faiss-gpu` (or `faiss-cpu` for fallback)."
    ) from exc


MAX_BATCH_LIMIT = 20_000_000


@dataclass
class ByteVecsInfo:
    path: Path
    num_vectors: int
    dim: int


@dataclass
class FloatVecsInfo:
    path: Path
    num_vectors: int
    dim: int


@dataclass
class Int8BinInfo:
    path: Path
    num_vectors: int
    dim: int


class FvecsMemmapReader:
    def __init__(self, path: Path, max_vectors: Optional[int] = None):
        self.info = inspect_fvecs(path, max_vectors=max_vectors)
        record_dtype = np.dtype([("dim", np.int32), ("vec", np.float32, (self.info.dim,))])
        self._mm = np.memmap(
            self.info.path,
            mode="r",
            dtype=record_dtype,
            shape=(self.info.num_vectors,),
        )

    @property
    def num_vectors(self) -> int:
        return self.info.num_vectors

    @property
    def dim(self) -> int:
        return self.info.dim

    def read_slice(self, start: int, end: int) -> np.ndarray:
        batch = np.asarray(self._mm[start:end]["vec"], dtype=np.float32)
        return np.ascontiguousarray(batch)

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        batch = np.asarray(self._mm[indices]["vec"], dtype=np.float32)
        return np.ascontiguousarray(batch)

    def iter_batches(self, batch_size: int) -> Iterator[Tuple[int, np.ndarray]]:
        for start in range(0, self.num_vectors, batch_size):
            end = min(start + batch_size, self.num_vectors)
            yield start, self.read_slice(start, end)


class BvecsMemmapReader:
    def __init__(self, path: Path, max_vectors: Optional[int] = None):
        self.info = inspect_bvecs(path, max_vectors=max_vectors)
        record_dtype = np.dtype([("dim", np.int32), ("vec", np.uint8, (self.info.dim,))])
        self._mm = np.memmap(
            self.info.path,
            mode="r",
            dtype=record_dtype,
            shape=(self.info.num_vectors,),
        )

    @property
    def num_vectors(self) -> int:
        return self.info.num_vectors

    @property
    def dim(self) -> int:
        return self.info.dim

    def read_slice(self, start: int, end: int) -> np.ndarray:
        batch = np.asarray(self._mm[start:end]["vec"], dtype=np.float32)
        return np.ascontiguousarray(batch)

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        batch = np.asarray(self._mm[indices]["vec"], dtype=np.float32)
        return np.ascontiguousarray(batch)

    def iter_batches(self, batch_size: int) -> Iterator[Tuple[int, np.ndarray]]:
        for start in range(0, self.num_vectors, batch_size):
            end = min(start + batch_size, self.num_vectors)
            yield start, self.read_slice(start, end)


class SpaceVBINMemmapReader:
    def __init__(self, path: Path, max_vectors: Optional[int] = None):
        self.info = inspect_spacev_bin(path, max_vectors=max_vectors)
        self._mm = np.memmap(
            self.info.path,
            mode="r",
            dtype=np.int8,
            offset=8,
            shape=(self.info.num_vectors, self.info.dim),
        )

    @property
    def num_vectors(self) -> int:
        return self.info.num_vectors

    @property
    def dim(self) -> int:
        return self.info.dim

    def read_slice(self, start: int, end: int) -> np.ndarray:
        batch = np.asarray(self._mm[start:end], dtype=np.float32)
        return np.ascontiguousarray(batch)

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        batch = np.asarray(self._mm[indices], dtype=np.float32)
        return np.ascontiguousarray(batch)

    def iter_batches(self, batch_size: int) -> Iterator[Tuple[int, np.ndarray]]:
        for start in range(0, self.num_vectors, batch_size):
            end = min(start + batch_size, self.num_vectors)
            yield start, self.read_slice(start, end)


def inspect_spacev_bin(path: Path, max_vectors: Optional[int] = None) -> Int8BinInfo:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Empty or invalid SpaceV .bin file: {path}")
        num_vectors, dim = struct.unpack("<ii", header)

    if num_vectors <= 0 or dim <= 0:
        raise ValueError(f"Invalid SpaceV .bin header: num_vectors={num_vectors}, dim={dim}")

    file_size = path.stat().st_size
    expected_size = 8 + int(num_vectors) * int(dim)
    if file_size != expected_size:
        raise ValueError(
            f"SpaceV .bin size mismatch: expected {expected_size} bytes, got {file_size} bytes for {path}"
        )

    if max_vectors is not None:
        num_vectors = min(num_vectors, int(max_vectors))

    return Int8BinInfo(path=path, num_vectors=num_vectors, dim=dim)


def inspect_bvecs(path: Path, max_vectors: Optional[int] = None) -> ByteVecsInfo:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("rb") as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError(f"Empty or invalid bvecs file: {path}")

    dim = int(np.frombuffer(raw, dtype=np.int32)[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension {dim} in {path}")

    record_bytes = 4 + dim
    file_size = path.stat().st_size
    if file_size % record_bytes != 0:
        raise ValueError(
            f"File size {file_size} is not divisible by record size {record_bytes}: {path}"
        )

    num_vectors = file_size // record_bytes
    if max_vectors is not None:
        num_vectors = min(num_vectors, int(max_vectors))

    return ByteVecsInfo(path=path, num_vectors=num_vectors, dim=dim)


def inspect_fvecs(path: Path, max_vectors: Optional[int] = None) -> FloatVecsInfo:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("rb") as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError(f"Empty or invalid fvecs file: {path}")

    dim = int(np.frombuffer(raw, dtype=np.int32)[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension {dim} in {path}")

    record_bytes = 4 + dim * 4
    file_size = path.stat().st_size
    if file_size % record_bytes != 0:
        raise ValueError(
            f"File size {file_size} is not divisible by record size {record_bytes}: {path}"
        )

    num_vectors = file_size // record_bytes
    if max_vectors is not None:
        num_vectors = min(num_vectors, int(max_vectors))

    return FloatVecsInfo(path=path, num_vectors=num_vectors, dim=dim)


def infer_data_format(path: Path, dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    suffix = path.suffix.lower()

    if dataset_name.startswith("spacev"):
        return "spacev-bin"
    if suffix == ".bvecs":
        return "bvecs"
    if suffix == ".fvecs":
        return "fvecs"
    if suffix == ".bin":
        return "spacev-bin"
    raise ValueError(
        f"Cannot infer data format from dataset_name={dataset_name!r}, path={str(path)!r}."
    )


def make_reader(path: Path, dataset_name: str, data_format: str, max_vectors: Optional[int]):
    if data_format == "auto":
        data_format = infer_data_format(path, dataset_name)

    if data_format == "spacev-bin":
        return SpaceVBINMemmapReader(path, max_vectors=max_vectors)
    if data_format == "bvecs":
        return BvecsMemmapReader(path, max_vectors=max_vectors)
    if data_format == "fvecs":
        return FvecsMemmapReader(path, max_vectors=max_vectors)

    raise ValueError(f"Unsupported --data-format: {data_format}")


def default_output_path(
    output_dir: Path,
    dataset_name: str,
    dataset_size_label: str,
    num_clusters: int,
    epsilon: float,
    min_replica: int,
    max_replica: int,
) -> Path:
    filename = (
        f"{dataset_name}{dataset_size_label}-{num_clusters}cluster-{epsilon:.3f}-min{min_replica}-max{max_replica}.bin"
    )
    return output_dir / filename


def create_search_index(dim: int, use_gpu: bool, device: int, tempmem: int):
    cpu_index = faiss.IndexFlatL2(dim)
    if not use_gpu:
        return cpu_index, False

    try:
        resources = faiss.StandardGpuResources()
        if tempmem > 0 and hasattr(resources, "setTempMemory"):
            resources.setTempMemory(int(tempmem))
        gpu_index = faiss.index_cpu_to_gpu(resources, device, cpu_index)
        return gpu_index, True
    except Exception as exc:
        print(f"[Warn] GPU Faiss index is unavailable, fallback to CPU index: {exc}")
        return cpu_index, False


def train_initial_centroids(
    sample: np.ndarray,
    num_clusters: int,
    use_gpu: bool,
    train_iterations: int,
    nredo: int,
    seed: int,
) -> np.ndarray:
    dim = sample.shape[1]
    kmeans = faiss.Kmeans(dim, num_clusters, niter=train_iterations, nredo=nredo, verbose=True, gpu=use_gpu)
    if hasattr(kmeans, "cp"):
        avg_points = max(1.0, float(sample.shape[0]) / float(num_clusters))
        kmeans.cp.seed = seed
        kmeans.cp.min_points_per_centroid = max(1, int(avg_points / 5.0))
        kmeans.cp.max_points_per_centroid = max(
            kmeans.cp.min_points_per_centroid + 1,
            int(avg_points * 5.0),
        )
    kmeans.train(sample)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(num_clusters, dim)
    return np.ascontiguousarray(centroids)


def compute_replica_counts(
    distances: np.ndarray,
    min_replica: int,
    max_replica: int,
    epsilon: float,
) -> np.ndarray:
    if min_replica == max_replica:
        return np.full(distances.shape[0], min_replica, dtype=np.int32)

    threshold = (1.0 + epsilon) * distances[:, :1]
    replica_counts = np.sum(distances <= threshold, axis=1, dtype=np.int32)
    replica_counts = np.maximum(replica_counts, min_replica)
    replica_counts = np.minimum(replica_counts, max_replica)
    return replica_counts.astype(np.int32, copy=False)


def search_with_duplication(
    index,
    batch: np.ndarray,
    min_replica: int,
    max_replica: int,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, labels = index.search(batch, max_replica)
    labels = labels.astype(np.int64, copy=False)
    replica_counts = compute_replica_counts(distances, min_replica, max_replica, epsilon)
    return distances, labels, replica_counts


def accumulate_grouped(
    vectors: np.ndarray,
    cluster_ids: np.ndarray,
    centroid_sums: np.ndarray,
    cluster_counts: np.ndarray,
) -> None:
    if cluster_ids.size == 0:
        return

    order = np.argsort(cluster_ids, kind="stable")
    cluster_ids_sorted = cluster_ids[order]
    vectors_sorted = vectors[order]
    unique_ids, first_pos, counts = np.unique(
        cluster_ids_sorted,
        return_index=True,
        return_counts=True,
    )
    centroid_sums[unique_ids] += np.add.reduceat(vectors_sorted, first_pos, axis=0)
    cluster_counts[unique_ids] += counts.astype(np.int64, copy=False)


def streamed_refine_centroids(
    reader: FvecsMemmapReader,
    centroids: np.ndarray,
    batch_size: int,
    min_replica: int,
    max_replica: int,
    epsilon: float,
    use_gpu: bool,
    device: int,
    tempmem: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    num_clusters, dim = centroids.shape
    index, gpu_enabled = create_search_index(dim, use_gpu, device, tempmem)
    index.add(centroids)

    centroid_sums = np.zeros((num_clusters, dim), dtype=np.float64)
    cluster_counts = np.zeros(num_clusters, dtype=np.int64)
    inertia = 0.0

    for start, batch in reader.iter_batches(batch_size):
        distances, labels, replica_counts = search_with_duplication(
            index, batch, min_replica, max_replica, epsilon
        )
        inertia += float(np.sum(distances[:, 0], dtype=np.float64))

        for replica_id in range(max_replica):
            mask = replica_counts > replica_id
            if not np.any(mask):
                continue
            accumulate_grouped(
                batch[mask],
                labels[mask, replica_id],
                centroid_sums,
                cluster_counts,
            )

        batch_end = start + batch.shape[0]
        print(
            f"[Refine] processed {batch_end:,}/{reader.num_vectors:,} "
            f"(gpu={'yes' if gpu_enabled else 'no'})"
        )

    new_centroids = centroids.copy()
    valid = cluster_counts > 0
    new_centroids[valid] = (centroid_sums[valid] / cluster_counts[valid, None]).astype(np.float32)
    return np.ascontiguousarray(new_centroids), cluster_counts, inertia


def count_cluster_members(
    reader: FvecsMemmapReader,
    centroids: np.ndarray,
    batch_size: int,
    min_replica: int,
    max_replica: int,
    epsilon: float,
    use_gpu: bool,
    device: int,
    tempmem: int,
) -> Tuple[np.ndarray, float]:
    num_clusters, dim = centroids.shape
    index, gpu_enabled = create_search_index(dim, use_gpu, device, tempmem)
    index.add(centroids)

    cluster_counts = np.zeros(num_clusters, dtype=np.uint64)
    inertia = 0.0

    for start, batch in reader.iter_batches(batch_size):
        distances, labels, replica_counts = search_with_duplication(
            index, batch, min_replica, max_replica, epsilon
        )
        inertia += float(np.sum(distances[:, 0], dtype=np.float64))

        for replica_id in range(max_replica):
            mask = replica_counts > replica_id
            if not np.any(mask):
                continue
            binc = np.bincount(labels[mask, replica_id], minlength=num_clusters)
            cluster_counts += binc.astype(np.uint64, copy=False)

        batch_end = start + batch.shape[0]
        print(
            f"[Count] processed {batch_end:,}/{reader.num_vectors:,} "
            f"(gpu={'yes' if gpu_enabled else 'no'})"
        )

    return cluster_counts, inertia


def fill_cluster_members(
    reader: FvecsMemmapReader,
    centroids: np.ndarray,
    offsets: np.ndarray,
    temp_members_path: Path,
    batch_size: int,
    min_replica: int,
    max_replica: int,
    epsilon: float,
    use_gpu: bool,
    device: int,
    tempmem: int,
) -> None:
    num_clusters, dim = centroids.shape
    total_members = int(offsets[-1])
    members = np.memmap(temp_members_path, mode="w+", dtype=np.int32, shape=(total_members,))
    cursors = offsets[:-1].astype(np.uint64, copy=True)

    index, gpu_enabled = create_search_index(dim, use_gpu, device, tempmem)
    index.add(centroids)

    for start, batch in reader.iter_batches(batch_size):
        batch_size_real = batch.shape[0]
        _, labels, replica_counts = search_with_duplication(
            index, batch, min_replica, max_replica, epsilon
        )
        vector_ids = np.arange(start, start + batch_size_real, dtype=np.int32)

        for replica_id in range(max_replica):
            mask = replica_counts > replica_id
            if not np.any(mask):
                continue

            assigned_clusters = labels[mask, replica_id].astype(np.int64, copy=False)
            assigned_ids = vector_ids[mask]
            order = np.argsort(assigned_clusters, kind="stable")
            clusters_sorted = assigned_clusters[order]
            ids_sorted = assigned_ids[order]
            unique_clusters, first_pos, counts = np.unique(
                clusters_sorted,
                return_index=True,
                return_counts=True,
            )

            for cluster_id, first, count in zip(unique_clusters, first_pos, counts):
                write_pos = int(cursors[cluster_id])
                members[write_pos : write_pos + int(count)] = ids_sorted[first : first + int(count)]
                cursors[cluster_id] += int(count)

        batch_end = start + batch.shape[0]
        print(
            f"[Fill] processed {batch_end:,}/{reader.num_vectors:,} "
            f"(gpu={'yes' if gpu_enabled else 'no'})"
        )

    members.flush()
    del members


def write_result_file(
    output_path: Path,
    centroids: np.ndarray,
    num_vectors: int,
    offsets: np.ndarray,
    temp_members_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_members = int(offsets[-1])

    with output_path.open("wb") as f:
        np.asarray([centroids.shape[0], centroids.shape[1], num_vectors], dtype=np.uint64).tofile(f)
        np.asarray(centroids, dtype=np.float32).tofile(f)
        np.asarray(offsets, dtype=np.uint64).tofile(f)

        members = np.memmap(temp_members_path, mode="r", dtype=np.int32, shape=(total_members,))
        chunk = 16_000_000
        for start in range(0, total_members, chunk):
            end = min(start + chunk, total_members)
            np.asarray(members[start:end], dtype=np.int32).tofile(f)
        del members


def build_sample(
    reader: FvecsMemmapReader,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sample_size >= reader.num_vectors:
        print("[Init] sample_size >= dataset size, use full dataset as training sample.")
        return reader.read_slice(0, reader.num_vectors)

    sample_indices = np.sort(rng.choice(reader.num_vectors, size=sample_size, replace=False))
    print(f"[Init] sampling {sample_size:,} vectors for centroid initialization...")
    return reader.read_indices(sample_indices)


def summarise_clusters(cluster_counts: np.ndarray) -> None:
    min_size = int(cluster_counts.min()) if cluster_counts.size else 0
    max_size = int(cluster_counts.max()) if cluster_counts.size else 0
    mean_size = float(cluster_counts.mean()) if cluster_counts.size else 0.0
    singleton = int(np.sum(cluster_counts == 1))
    sorted_sizes = np.sort(cluster_counts)

    print("=== Summary ===")
    print(f"clusters: {cluster_counts.size}")
    print(f"cluster size min/max/avg: {min_size} / {max_size} / {mean_size:.2f}")
    print(f"singleton clusters: {singleton}")
    if sorted_sizes.size:
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            idx = min(int(q * sorted_sizes.size), sorted_sizes.size - 1)
            print(f"p{int(q * 100):02d}: {int(sorted_sizes[idx])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faiss-gpu clustering for large datasets in streamed batches.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("/mnt/anns-dataset/bigann/bigann_base.bvecs"),
        help="Path to the source dataset file.",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        choices=("auto", "fvecs", "bvecs", "spacev-bin"),
        default="auto",
        help="Input file format. Default: infer from path / dataset name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/anns-dataset/kmeans-output"),
        help="Directory for the clustering result file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output file path.",
    )
    parser.add_argument("--dataset-name", type=str, default="sift")
    parser.add_argument("--dataset-size-label", type=str, default="500M")
    parser.add_argument("--num-clusters", type=int, default=4096)
    parser.add_argument("--min-replica", type=int, default=1)
    parser.add_argument("--max-replica", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--train-sample-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--train-iterations", type=int, default=25)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--nredo", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10_000_000)
    parser.add_argument(
        "--max-vectors", type=int, default=500_000_000,
        help="Read at most this many vectors from the start of the dataset.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-temp-mem", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true", help="Use CPU Faiss only.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_clusters <= 0:
        raise ValueError("--num-clusters must be positive.")
    if args.min_replica <= 0:
        raise ValueError("--min-replica must be positive.")
    if args.max_replica < args.min_replica:
        raise ValueError("--max-replica must be >= --min-replica.")
    if args.batch_size <= 0 or args.batch_size > MAX_BATCH_LIMIT:
        raise ValueError(f"--batch-size must be in [1, {MAX_BATCH_LIMIT}].")
    if args.train_sample_size < args.num_clusters:
        raise ValueError("--train-sample-size must be >= --num-clusters.")
    if args.refine_iterations < 0:
        raise ValueError("--refine-iterations must be >= 0.")
    if args.max_vectors is not None and args.max_vectors <= 0:
        raise ValueError("--max-vectors must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    reader = make_reader(
        args.data_file,
        args.dataset_name,
        args.data_format,
        args.max_vectors,
    )
    output_path = args.output_file
    if output_path is None:
        output_path = default_output_path(
            args.output_dir,
            args.dataset_name,
            args.dataset_size_label,
            args.num_clusters,
            args.epsilon,
            args.min_replica,
            args.max_replica,
        )

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite it."
        )

    if reader.num_vectors < args.num_clusters:
        raise ValueError(
            f"Dataset size {reader.num_vectors} is smaller than num_clusters {args.num_clusters}."
        )

    print("=== Faiss Streamed Clustering ===")
    print(f"data file: {reader.info.path}")
    print(f"output file: {output_path}")
    print(f"vectors: {reader.num_vectors:,}")
    print(f"dimension: {reader.dim}")
    print(f"clusters: {args.num_clusters}")
    print(f"replica range: min:{args.min_replica}, max:{args.max_replica}")
    print(f"epsilon: {args.epsilon}")
    print(f"data format: {infer_data_format(args.data_file, args.dataset_name) if args.data_format == 'auto' else args.data_format}")
    print(f"max vectors: {args.max_vectors:,}" if args.max_vectors is not None else "max vectors: full dataset")
    print(f"batch size: {args.batch_size:,}")
    print(f"device: {'cpu' if args.cpu else f'gpu:{args.device}'}")
    print()

    sample_size = min(args.train_sample_size, reader.num_vectors)
    train_sample = build_sample(reader, sample_size, args.seed)
    centroids = train_initial_centroids(
        train_sample,
        args.num_clusters,
        use_gpu=not args.cpu,
        train_iterations=args.train_iterations,
        nredo=args.nredo,
        seed=args.seed,
    )
    del train_sample

    for refine_round in range(args.refine_iterations):
        print(f"\n=== Full refine pass {refine_round + 1}/{args.refine_iterations} ===")
        new_centroids, refine_counts, inertia = streamed_refine_centroids(
            reader=reader,
            centroids=centroids,
            batch_size=args.batch_size,
            min_replica=args.min_replica,
            max_replica=args.max_replica,
            epsilon=args.epsilon,
            use_gpu=not args.cpu,
            device=args.device,
            tempmem=args.gpu_temp_mem,
        )
        shift = float(np.linalg.norm(new_centroids - centroids, axis=1).mean())
        empty_clusters = int(np.sum(refine_counts == 0))
        print(
            f"[Refine] inertia={inertia:.6e}, mean centroid shift={shift:.6f}, "
            f"empty clusters={empty_clusters}"
        )
        centroids = new_centroids

    print("\n=== Counting final cluster sizes ===")
    cluster_counts, final_inertia = count_cluster_members(
        reader=reader,
        centroids=centroids,
        batch_size=args.batch_size,
        min_replica=args.min_replica,
        max_replica=args.max_replica,
        epsilon=args.epsilon,
        use_gpu=not args.cpu,
        device=args.device,
        tempmem=args.gpu_temp_mem,
    )
    offsets = np.zeros(args.num_clusters + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(cluster_counts, dtype=np.uint64)
    print(f"[Count] final inertia={final_inertia:.6e}")
    summarise_clusters(cluster_counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path_raw = tempfile.mkstemp(
        prefix=output_path.stem + ".members.",
        suffix=".tmp",
        dir=args.output_dir,
    )
    os.close(fd)
    temp_members_path = Path(temp_path_raw)

    try:
        print("\n=== Writing cluster membership ===")
        fill_cluster_members(
            reader=reader,
            centroids=centroids,
            offsets=offsets,
            temp_members_path=temp_members_path,
            batch_size=args.batch_size,
            min_replica=args.min_replica,
            max_replica=args.max_replica,
            epsilon=args.epsilon,
            use_gpu=not args.cpu,
            device=args.device,
            tempmem=args.gpu_temp_mem,
        )

        print("\n=== Saving binary result ===")
        write_result_file(
            output_path=output_path,
            centroids=centroids,
            num_vectors=reader.num_vectors,
            offsets=offsets,
            temp_members_path=temp_members_path,
        )
    finally:
        if temp_members_path.exists():
            temp_members_path.unlink()

    avg_dist = float(np.sqrt(final_inertia / float(reader.num_vectors)))
    print("\n=== Done ===")
    print(f"saved: {output_path}")
    print(f"average distance to centroid: {avg_dist:.6f}")


if __name__ == "__main__":
    main()
