import json
import glob

files = sorted(glob.glob("results/*.json"))
print(f"Total result json files: {len(files)}")
data = {}
for f in files:
    with open(f) as fp:
        d = json.load(fp)
        exp = d.get("experiment")
        ds = d.get("dataset")
        m = d.get("model")
        data.setdefault(exp, {}).setdefault(ds, {})[m] = d.get("results")

for exp, datasets in sorted(data.items(), key=lambda x: str(x[0])):
    print(f"\n=== Experiment: {exp} ===")
    for ds, models in sorted(datasets.items()):
        print(f"  Dataset: {ds}")
        for m, res in sorted(models.items(), key=lambda x: str(x[0])):
            if not isinstance(res, dict):
                continue
            if exp == "exp1_baselines":
                s = ", ".join(f"{k}={v.get('accuracy_mean', 0)*100:.1f}%" for k, v in res.items() if isinstance(v, dict))
                print(f"    Model {m}: {s}")
            elif exp == "exp5_formatting":
                s = ", ".join(f"{k}={v.get('accuracy', 0)*100:.1f}%" for k, v in res.items() if isinstance(v, dict))
                print(f"    Model {m}: {s}")
            elif exp == "exp4_ordering":
                s = ", ".join(f"{k}={v.get('accuracy', 0)*100:.1f}%" for k, v in res.items() if isinstance(v, dict))
                print(f"    Model {m}: {s}")
            elif exp == "exp2_k_sweep":
                parts = []
                for meth, vals in res.items():
                    if isinstance(vals, dict):
                        sub = ", ".join(f"k{k}={v.get('accuracy', 0)*100:.1f}%" for k, v in vals.items() if isinstance(v, dict))
                        parts.append(f"{meth}: [{sub}]")
                print(f"    Model {m}: " + " | ".join(parts))
