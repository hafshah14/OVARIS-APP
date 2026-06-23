import joblib

bundle = joblib.load(
    "models/ovaris_final_full_model.joblib"
)

print(bundle.keys())

print("\nFEATURES:")
for i, f in enumerate(bundle["features"], 1):
    print(i, f)