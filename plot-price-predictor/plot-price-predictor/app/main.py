"""FastAPI service: plot price estimation with model comparison.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from pathlib import Path

from app.model_loader import MAPE, ModelStore
from app.schemas import ModelName, PlotFeatures, PredictionResponse

store: ModelStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = ModelStore()          # load artifacts once, not per request
    yield


app = FastAPI(
    title="Plot Price Predictor",
    description="ML/DL API that estimates land plot prices from size (acres), "
                "location, zoning, road access and frontage. Serves four models "
                "(Ridge baseline, Random Forest, Gradient Boosting, PyTorch MLP) "
                "and returns the ensemble comparison.",
    version="1.0.0",
    lifespan=lifespan,
)

RECOMMENDED: ModelName = "neural_net"   # best test MAPE (see models/metrics.json)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": store is not None}


@app.get("/models")
def models():
    return {
        "available": list(MAPE.keys()),
        "test_mape": MAPE,
        "recommended": RECOMMENDED,
        "features": {
            "numeric": ["acres", "distance_to_city_km", "frontage_m"],
            "categorical": ["location", "zoning", "road_access", "corner_plot"],
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PlotFeatures):
    preds = store.predict(features.model_dump())
    best = preds[RECOMMENDED]
    mape = MAPE[RECOMMENDED]
    return PredictionResponse(
        input=features,
        predictions={k: round(v, 0) for k, v in preds.items()},
        recommended_model=RECOMMENDED,
        estimated_price=round(best, 0),
        price_range={
            "low": round(best * (1 - mape), 0),
            "high": round(best * (1 + mape), 0),
        },
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


from app.portfolio_store import session_id, save_run, list_runs, get_run

@app.get('/runs')
def history(session:str=Depends(session_id)):
    return list_runs(session)

@app.post('/runs')
def tracked_prediction(features:PlotFeatures,session:str=Depends(session_id)):
    return save_run(session,'estimate',{'records':[predict(features).model_dump(mode='json')]})

class ScenarioComparison(BaseModel):
    baseline:PlotFeatures
    alternative:PlotFeatures

@app.post('/compare')
def compare(body:ScenarioComparison,session:str=Depends(session_id)):
    a=predict(body.baseline).model_dump(mode='json');b=predict(body.alternative).model_dump(mode='json')
    delta=b['estimated_price']-a['estimated_price']
    return save_run(session,'comparison',{'records':[a,b],'difference':delta,'change_percent':round(100*delta/a['estimated_price'],2) if a['estimated_price'] else None})

@app.post('/runs/csv')
def estimate_csv(file:UploadFile=File(...),session:str=Depends(session_id)):
    import csv,io
    raw=file.file.read(1024*1024+1)
    if len(raw)>1024*1024:raise HTTPException(413,'CSV exceeds 1 MB')
    try:
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        if not reader.fieldnames or not {'acres','location'}.issubset(reader.fieldnames):raise ValueError('CSV needs acres and location columns')
        if set(reader.fieldnames)-set(PlotFeatures.model_fields):raise ValueError('Unrecognized property columns')
        records=[]
        for n,row in enumerate(reader,2):
            if n>101:raise ValueError('Maximum 100 properties per batch')
            if None in row or any(v is None for v in row.values()):raise ValueError(f'Row {n}: inconsistent number of fields')
            try:features=PlotFeatures.model_validate({k:v for k,v in row.items() if v.strip()!=''})
            except ValueError:raise ValueError(f'Row {n}: invalid property details')
            records.append(predict(features).model_dump(mode='json'))
        if not records:raise ValueError('CSV has no properties')
    except (ValueError,UnicodeDecodeError,csv.Error) as e:raise HTTPException(422,str(e))
    return save_run(session,'batch',{'records':records})

@app.get('/sample.csv')
def sample_csv():
    return Response('acres,location,zoning,road_access,corner_plot,distance_to_city_km,frontage_m\n2.5,Lakeview,residential,paved,true,4,30\n1.2,Downtown,commercial,paved,false,2,25\n4.5,Greenfield,agricultural,gravel,false,20,50\n',media_type='text/csv',headers={'Content-Disposition':'attachment; filename="plotlens-synthetic-sample.csv"'})

@app.get('/runs/{run_id}/report.csv')
def report_csv(run_id:str,session:str=Depends(session_id)):
    import csv,io
    run=get_run(session,run_id)
    if not run:raise HTTPException(404,'Run not found in this browser session')
    out=io.StringIO();w=csv.writer(out);w.writerow(['synthetic_location','acres','ridge','random_forest','gradient_boosting','neural_net','heuristic_low','heuristic_high','currency'])
    for r in run['data']['records']:w.writerow([r['input']['location'],r['input']['acres'],*[r['predictions'][m] for m in ['ridge','random_forest','gradient_boosting','neural_net']],r['price_range']['low'],r['price_range']['high'],'USD'])
    return Response(out.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="plotlens-synthetic-estimates.csv"'})
