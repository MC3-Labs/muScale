# muScale report

- Confusion matrix: `food101/outputs/confusion_vitb32.csv`
- K classes: 101
- n samples (in confusion): 500
- Overall weak-label accuracy (on sampled gold set): 0.8620

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
- beef_carpaccio: acc=0.500 (n=4)
- donuts: acc=0.500 (n=4)
- hot_dog: acc=0.500 (n=2)
- macaroni_and_cheese: acc=0.500 (n=2)
- paella: acc=0.500 (n=2)
- steak: acc=0.500 (n=8)
- tuna_tartare: acc=0.500 (n=2)
- escargots: acc=0.600 (n=5)
- hummus: acc=0.600 (n=5)
- strawberry_shortcake: acc=0.600 (n=5)
- waffles: acc=0.600 (n=10)
- apple_pie: acc=0.667 (n=3)
- ceviche: acc=0.667 (n=3)
