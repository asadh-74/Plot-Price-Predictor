# AWS Deployment Guide

## Option A — App Runner (recommended for portfolio)
Fully managed, deploys straight from GitHub, auto-rebuilds on push, free tier friendly.

1. Push repo to GitHub.
2. AWS Console → App Runner → Create service → Source: GitHub → pick repo.
3. Build settings: **Dockerfile** (root). Port 8000.
4. Create & deploy. You get an HTTPS URL like `https://xxx.us-east-1.awsapprunner.com`.

## Option B — EC2 + Docker

```bash
sudo apt update && sudo apt install -y docker.io
git clone <your-repo> && cd plot-price-predictor
docker build -t plot-price .
docker run -d -p 80:8000 --restart unless-stopped plot-price
```
Point an Elastic IP + security group (port 80/443) at it.

## Option C — SageMaker (for client-scale work)
Wrap `ModelStore.predict` in a `model_fn`/`predict_fn` inference script, push the
`models/` artifacts to S3, deploy a real-time endpoint. Overkill for demo traffic.

## Laravel integration

```php
$response = Http::post('https://YOUR-APP-RUNNER-URL/predict', [
    'acres' => $listing->acres, 'location' => $listing->area,
    'zoning' => $listing->zoning, 'road_access' => $listing->road_access,
    'corner_plot' => $listing->is_corner, 'distance_to_city_km' => $listing->dist_km,
    'frontage_m' => $listing->frontage,
]);
$suggested = $response->json()['estimated_price']; // show as "AI-suggested price"
```
