# muScale bundle report

- Dataset: Food101
- Weak labeler: open_clip ViT-B-32 (laion2b_s34b_b79k), prompt=food101/prompts.py
- Confusion: `food101/outputs/confusion_vitb32.csv`
- K classes: 101
- n samples (in confusion): 500
- Overall weak-label accuracy (on sampled gold set): 0.8620

## muScale summary
- H(Y) (bits): 6.4864
- I(Y;Y~) (bits): 5.9723
- lambda_pred = I/H: 0.9207

Interpretation (random-error regime assumption):

- 1 weak label ~= 0.921 gold labels

## Top confusions (count; percent within true class)
- ice_cream → frozen_yogurt: 3 (75.0% of that true class)
- steak → pork_chop: 3 (37.5% of that true class)
- apple_pie → bread_pudding: 1 (33.3% of that true class)
- beef_carpaccio → hot_and_sour_soup: 1 (25.0% of that true class)
- beef_carpaccio → ramen: 1 (25.0% of that true class)
- beef_tartare → beef_carpaccio: 1 (20.0% of that true class)
- beet_salad → cheese_plate: 1 (12.5% of that true class)
- beet_salad → peking_duck: 1 (12.5% of that true class)
- beignets → cannoli: 1 (10.0% of that true class)
- beignets → french_toast: 1 (10.0% of that true class)
- bruschetta → escargots: 1 (33.3% of that true class)
- bruschetta → grilled_salmon: 1 (33.3% of that true class)
- cannoli → croque_madame: 1 (9.1% of that true class)
- caprese_salad → bruschetta: 1 (11.1% of that true class)
- ceviche → beef_carpaccio: 1 (33.3% of that true class)

## Lowest per-class accuracy (quick scan)
- ice_cream: acc=0.250 (n=4)
- bruschetta: acc=0.333 (n=3)
- donuts: acc=0.500 (n=4)
- beef_carpaccio: acc=0.500 (n=4)
- hot_dog: acc=0.500 (n=2)
- macaroni_and_cheese: acc=0.500 (n=2)
- steak: acc=0.500 (n=8)
- paella: acc=0.500 (n=2)
- tuna_tartare: acc=0.500 (n=2)
- strawberry_shortcake: acc=0.600 (n=5)
- escargots: acc=0.600 (n=5)
- waffles: acc=0.600 (n=10)
- hummus: acc=0.600 (n=5)
- apple_pie: acc=0.667 (n=3)
- greek_salad: acc=0.667 (n=3)

## Artifacts
- JSON: `vitb32_bundle.json`
- Per-class CSV: `vitb32_bundle_per_class.csv`
- Heatmap: `vitb32_bundle_heatmap.png`
