import joblib
import pprint

model = joblib.load('best_model.pkl')
print('Model type:', type(model))
try:
    print('n_features_in_:', getattr(model, 'n_features_in_', None))
except Exception as e:
    print('n_features_in_ error:', e)

# For pipelines, inspect final estimator
from sklearn.pipeline import Pipeline
if isinstance(model, Pipeline):
    print('\nPipeline steps:')
    for name, step in model.steps:
        print('-', name, type(step))
    final = model.steps[-1][1]
    print('\nFinal estimator type:', type(final))
    print('Final n_features_in_:', getattr(final, 'n_features_in_', None))
    print('Final feature_names_in_:', getattr(final, 'feature_names_in_', None))
else:
    print('feature_names_in_:', getattr(model, 'feature_names_in_', None))

# Try to print any training column names stored in the model
for attr in ['feature_names_in_', 'columns_', 'feature_names', 'feature_names_out_']:
    if hasattr(model, attr):
        print(f"{attr}:", getattr(model, attr))

# If the model is a sk-learn object with get_params, print top-level keys
try:
    pp = pprint.PrettyPrinter(indent=2)
    params = model.get_params()
    print('\nTop-level params keys:', list(params.keys())[:40])
except Exception as e:
    print('get_params error:', e)
