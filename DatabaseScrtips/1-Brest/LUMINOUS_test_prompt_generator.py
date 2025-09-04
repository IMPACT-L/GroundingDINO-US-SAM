#%%
import csv
import random
import glob
# Base templates for prompts
longitudinal_templates = [
    "Longitudinal ultrasound of the lumbar multifidus showing {feature}.",
    "Longitudinal view of the LM muscle with {feature}.",
    "LM muscle in longitudinal section, displaying {feature}."
]

transverse_templates = [
    "Transverse ultrasound scan of the lumbar multifidus with {feature}.",
    "Transverse view of the LM muscle showing {feature}.",
    "Cross-sectional ultrasound of the LM muscle, {feature}."
]

# Possible muscle features
features = [
    "parallel muscle fibers and hyperechoic perimysium",
    "visible fascicles and hypoechoic muscle tissue",
    "uniform echogenicity and fascial boundaries",
    "muscle-tendon junctions and adjacent connective tissue",
    "cross-sectional area and surrounding aponeurosis",
    "tendon attachments and minimal fatty infiltration",
    "muscle striations and echogenic septa",
    "well-defined borders and homogeneous muscle texture",
    "fascicle orientation and adjacent vertebral structures",
    "mild fatty infiltration and disrupted fiber pattern",
    "manual segmentation overlay for deep learning",
    "reduced muscle thickness due to atrophy",
    "optimized depth settings for clear LM visualization",
    "high-frequency details of muscle fascicles"
]

#%% Generate 341 unique prompts
names = [name.split('/')[-1] for name in glob.glob('../Dataset/LUMINOUS_Database/B-mode/*')]
#%%
prompts = []
for image_name in names:    
    # Alternate between longitudinal and transverse views
    if i % 2 == 0:
        template = random.choice(longitudinal_templates)
    else:
        template = random.choice(transverse_templates)
    
    feature = random.choice(features)
    prompt = template.format(feature=feature)
    
    prompts.append([image_name, prompt])

# Save to CSV
with open("luminous_prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "prompt_text"])
    writer.writerows(prompts)

print("CSV generated: luminous_prompts.csv")
# %%
