import pandas as pd
from beans import extract_features

pt_model = pd.read_pickle("nasnetmobile.model")

image_features = extract_features(pt_model = pt_model,
                                  img_path = "dataset/val/bag2.jpg",
                                  single_image = True)

stored_vars = pd.read_pickle("vars.dict")
prediction = list(stored_vars["model"].predict(image_features)[0]) #we have just one element

def get_category(prediction):
    max_proba = max(prediction)
    if max_proba < 0.5:
        return "Don't know"
    else:
        correspondances = stored_vars["correspondances"]
        return correspondances[prediction.index(max_proba)]

category = get_category(prediction)
