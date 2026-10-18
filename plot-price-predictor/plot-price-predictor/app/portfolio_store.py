"""Session-scoped portfolio run history. Use a durable DB_PATH for persistent hosting."""
import json, os, re, sqlite3, uuid
from datetime import datetime, timezone
from fastapi import Request, Response, HTTPException

DB_PATH=os.getenv('DEMO_HISTORY_DB', '/tmp/plotlens-history.db')
def connection():
    db=sqlite3.connect(DB_PATH, timeout=20)
    db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, session TEXT NOT NULL, created TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS runs_session ON runs(session, created)')
    return db

def session_id(request:Request,response:Response):
    session=request.cookies.get('portfolio_session','')
    if not re.fullmatch(r'[a-f0-9]{32}',session):
        session=uuid.uuid4().hex
        response.set_cookie('portfolio_session',session,httponly=True,samesite='lax',secure=request.url.scheme=='https',max_age=86400)
    return session

def save_run(session,kind,data):
    item={'id':uuid.uuid4().hex,'created':datetime.now(timezone.utc).isoformat(),'kind':kind,'data':data}
    with connection() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',(item['id'],session,item['created'],kind,json.dumps(data,allow_nan=False)))
        db.execute('DELETE FROM runs WHERE session=? AND id NOT IN (SELECT id FROM runs WHERE session=? ORDER BY created DESC LIMIT 20)',(session,session))
    return item

def list_runs(session):
    with connection() as db:
        rows=db.execute('SELECT id,created,kind,data FROM runs WHERE session=? ORDER BY created DESC',(session,)).fetchall()
    return [{'id':r[0],'created':r[1],'kind':r[2],'data':json.loads(r[3])} for r in rows]

def get_run(session,run_id):
    return next((r for r in list_runs(session) if r['id']==run_id),None)
