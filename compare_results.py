import numpy as np
import struct
import os
import glob
import argparse

def must_exist(path):
    if not os.path.exists(path):
        nearby = "\n  ".join(sorted(glob.glob("*.ds2")))
        raise FileNotFoundError(
            f"File non trovato: {path}\n\nFile .ds2 presenti nella cartella:\n  {nearby}"
        )

def load_ds2_auto(filename, kind):
    """
    kind: "int" oppure "float"
    Auto-detect tra 32/64 bit (4 o 8 byte) dai byte residui nel file.
    Formato ds2: [int32 n][int32 d][data...]
    """
    must_exist(filename)

    filesize = os.path.getsize(filename)
    with open(filename, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        d = struct.unpack("i", f.read(4))[0]

        elems = n * d
        data_bytes = filesize - 8

        if elems <= 0:
            raise ValueError(f"Header non valido in {filename}: n={n}, d={d}")
        if data_bytes % elems != 0:
            raise ValueError(
                f"Dimensione file incoerente in {filename}: "
                f"bytes dati={data_bytes}, elems={elems}"
            )

        bpe = data_bytes // elems  # bytes per elemento: 4 o 8
        if kind == "float":
            if bpe == 4:
                dtype = np.float32
            elif bpe == 8:
                dtype = np.float64
            else:
                raise ValueError(f"Bytes/elem non supportati ({bpe}) in {filename}")
        elif kind == "int":
            if bpe == 4:
                dtype = np.int32
            elif bpe == 8:
                dtype = np.int64
            else:
                raise ValueError(f"Bytes/elem non supportati ({bpe}) in {filename}")
        else:
            raise ValueError("kind deve essere 'int' o 'float'")

        data = np.fromfile(f, dtype=dtype).reshape(n, d)
        return data, dtype

def compare_variant(label, ref_bits, out_bits,
                    N=2000, D=256, nq=2000, k=8, x=64,
                    dist_tol=0.2, show_first=10):

    # Ground truth (prof)
    prof_ids = f"results_ids_{nq}x{k}_k{k}_x{x}_{ref_bits}.ds2"
    prof_dst = f"results_dst_{nq}x{k}_k{k}_x{x}_{ref_bits}.ds2"

    # Output tuoi
    my_ids = f"idNN_{out_bits}_size-{N}x{D}_nq-{nq}.ds2"
    my_dst = f"distNN_{out_bits}_size-{N}x{D}_nq-{nq}.ds2"

    id_prof, id_prof_dt = load_ds2_auto(prof_ids, "int")
    dist_prof, dist_prof_dt = load_ds2_auto(prof_dst, "float")
    id_me, id_me_dt = load_ds2_auto(my_ids, "int")
    dist_me, dist_me_dt = load_ds2_auto(my_dst, "float")

    print("\n" + "="*70)
    print(f"CONFRONTO = {label}   (ref={ref_bits} vs out={out_bits})")
    print(f"Prof IDs : {prof_ids}  dtype={id_prof_dt.__name__}")
    print(f"Prof DST : {prof_dst}  dtype={dist_prof_dt.__name__}")
    print(f"Tuoi IDs : {my_ids}    dtype={id_me_dt.__name__}")
    print(f"Tuoi DST : {my_dst}    dtype={dist_me_dt.__name__}")
    print("="*70)

    # IDs
    print("Confronto ID...")
    ids_ok = np.array_equal(id_prof, id_me)
    print("ID identici:", ids_ok)

    if not ids_ok:
        diff = np.where(id_prof != id_me)
        print("Numero differenze:", len(diff[0]))
        print("Prime differenze (prof -> tuo):")
        for idx in range(min(show_first, len(diff[0]))):
            i, j = diff[0][idx], diff[1][idx]
            print(f"  posizione {i},{j}: prof={id_prof[i,j]}  tu={id_me[i,j]}")

    # DIST (confronto in float64 per sicurezza)
    print("\nConfronto distanze...")
    dp = dist_prof.astype(np.float64, copy=False)
    dm = dist_me.astype(np.float64, copy=False)
    abs_diff = np.abs(dp - dm)

    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print("Massima differenza:", max_diff)
    print("Differenza media:", mean_diff)

    dist_ok = max_diff <= dist_tol
    print("Distanze compatibili:", dist_ok, f"(tol={dist_tol})")

    return ids_ok, dist_ok

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronta i risultati con la ground truth del prof.")
    parser.add_argument(
        "--t", required=True, choices=["32", "64", "64omp"],
        help="Variante da confrontare (deve corrispondere a quella che hai appena runnato)."
    )
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--nq", type=int, default=2000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--x", type=int, default=64)
    parser.add_argument("--dist_tol", type=float, default=0.2)
    parser.add_argument("--show_first", type=int, default=10)

    args = parser.parse_args()

    # mapping: quali file di riferimento e output usare
    if args.t == "32":
        ref_bits, out_bits = "32", "32"
    elif args.t == "64":
        ref_bits, out_bits = "64", "64"
    else:  # "64omp"
        ref_bits, out_bits = "64", "64"  # i file sono gli stessi del 64

    compare_variant(
        label=args.t,
        ref_bits=ref_bits,
        out_bits=out_bits,
        N=args.N, D=args.D, nq=args.nq, k=args.k, x=args.x,
        dist_tol=args.dist_tol,
        show_first=args.show_first
    )