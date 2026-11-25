"""SQLite layer: users, jobs, applications, notifications, compact chat memory, prefs."""
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
CV_DIR = os.path.join(DATA, "cvs")
DB_PATH = os.path.join(DATA, "app.db")
os.makedirs(CV_DIR, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, role TEXT, salt TEXT, pw TEXT);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, hr_id INTEGER, title TEXT, description TEXT, skills TEXT,
  min_years REAL DEFAULT 0, status TEXT DEFAULT 'open', thread TEXT, ts REAL,
  extras TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS apps(
  id INTEGER PRIMARY KEY, job_id INTEGER, cand_id INTEGER, cv_hash TEXT,
  status TEXT DEFAULT 'draft', final REAL, eval TEXT, ts REAL,
  UNIQUE(job_id, cand_id));
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, user_id INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS prefs(user_id INTEGER, k TEXT, v TEXT, PRIMARY KEY(user_id, k));
CREATE TABLE IF NOT EXISTS msgs(
  id INTEGER PRIMARY KEY, user_id INTEGER, scope TEXT, role TEXT, content TEXT);
CREATE TABLE IF NOT EXISTS notifs(
  id INTEGER PRIMARY KEY, user_id INTEGER, job_id INTEGER, text TEXT, seen INTEGER DEFAULT 0,
  ts REAL, UNIQUE(user_id, job_id));
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def q(sql, args=(), one=False):
    with conn() as c:
        rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    return (rows[0] if rows else None) if one else rows


def ex(sql, args=()):
    with conn() as c:
        return c.execute(sql, args).lastrowid


# ---------- auth ----------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 120_000).hex()


def register(email, name, pw, role):
    email = email.strip().lower()
    name = name.strip()
    if not EMAIL_RE.match(email):
        return None, "Invalid email address."
    if len(pw) < 6:
        return None, "Password must be at least 6 characters."
    if not name:
        return None, "Name is required."
    is_hr_mail = email.endswith("@hr.com")
    if role == "hr" and not is_hr_mail:
        return None, "HR accounts must use an email ending with @hr.com."
    if role == "candidate" and is_hr_mail:
        return None, "@hr.com emails are reserved for HR accounts."
    if q("SELECT 1 FROM users WHERE email=?", (email,), one=True):
        return None, "This email is already registered."
    salt = os.urandom(16).hex()
    uid = ex("INSERT INTO users(email,name,role,salt,pw) VALUES(?,?,?,?,?)",
             (email, name, role, salt, _hash(pw, salt)))
    return get_user(uid), None


def login(email, pw):
    u = q("SELECT * FROM users WHERE email=?", (email.strip().lower(),), one=True)
    if not u or not hmac.compare_digest(u["pw"], _hash(pw, u["salt"])):
        return None
    return {k: u[k] for k in ("id", "email", "name", "role")}


def get_user(uid):
    return q("SELECT id,email,name,role FROM users WHERE id=?", (uid,), one=True)


# ---------- jobs ----------
def add_job(hr_id, title, description, skills, min_years, extras=None):
    return ex("INSERT INTO jobs(hr_id,title,description,skills,min_years,ts,extras) VALUES(?,?,?,?,?,?,?)",
              (hr_id, title, description, json.dumps(skills), float(min_years or 0), time.time(),
               json.dumps(extras or [], ensure_ascii=False)))


def update_job(job_id, hr_id, title, description, skills, min_years, extras):
    """Ownership-checked edit: only the HR who posted the job can change it."""
    ex("UPDATE jobs SET title=?, description=?, skills=?, min_years=?, extras=? WHERE id=? AND hr_id=?",
       (title, description, json.dumps(skills), float(min_years or 0),
        json.dumps(extras or [], ensure_ascii=False), job_id, hr_id))


def _job(r):
    if r:
        r["skills"] = json.loads(r["skills"] or "[]")
        r["extras"] = json.loads(r.get("extras") or "[]")
    return r


def get_job(job_id):
    return _job(q("SELECT j.*, u.name AS hr_name FROM jobs j JOIN users u ON u.id=j.hr_id WHERE j.id=?",
                  (job_id,), one=True))


def hr_job(job_id, hr_id):
    """Ownership-checked access: an HR can only ever read their own jobs."""
    return _job(q("SELECT * FROM jobs WHERE id=? AND hr_id=?", (job_id, hr_id), one=True))


def hr_jobs(hr_id):
    return [_job(r) for r in q("SELECT * FROM jobs WHERE hr_id=? ORDER BY id DESC", (hr_id,))]


def open_jobs():
    return [_job(r) for r in q(
        "SELECT j.*, u.name AS hr_name FROM jobs j JOIN users u ON u.id=j.hr_id "
        "WHERE j.status='open' ORDER BY j.id DESC")]


def set_job_status(job_id, hr_id, status):
    ex("UPDATE jobs SET status=? WHERE id=? AND hr_id=?", (status, job_id, hr_id))


def set_thread(job_id, thread):
    ex("UPDATE jobs SET thread=? WHERE id=?", (thread, job_id))


# ---------- applications ----------
def cv_path(job_id, cand_id):
    return os.path.join(CV_DIR, f"{job_id}_{cand_id}.pdf")


def get_app(job_id, cand_id):
    return q("SELECT * FROM apps WHERE job_id=? AND cand_id=?", (job_id, cand_id), one=True)


def save_draft(job_id, cand_id, cv_hash):
    """Insert or replace a DRAFT. A confirmed application can never be replaced."""
    a = get_app(job_id, cand_id)
    if a and a["status"] == "confirmed":
        return False
    if a:
        ex("UPDATE apps SET cv_hash=?, eval=NULL, final=NULL, ts=? WHERE id=?", (cv_hash, time.time(), a["id"]))
    else:
        ex("INSERT INTO apps(job_id,cand_id,cv_hash,ts) VALUES(?,?,?,?)", (job_id, cand_id, cv_hash, time.time()))
    return True


def save_eval(app_id, ev, final):
    ex("UPDATE apps SET eval=?, final=? WHERE id=?", (json.dumps(ev), final, app_id))


def confirm_app(app_id):
    ex("UPDATE apps SET status='confirmed' WHERE id=? AND status='draft' AND eval IS NOT NULL", (app_id,))


def cand_apps(cand_id):
    return q("SELECT a.*, j.title, u.name AS hr_name FROM apps a JOIN jobs j ON j.id=a.job_id "
             "JOIN users u ON u.id=j.hr_id WHERE a.cand_id=? ORDER BY a.id DESC", (cand_id,))


def job_applicants(job_id, hr_id):
    """Only confirmed applications of a job that belongs to this HR, best score first."""
    rows = q("SELECT a.id, a.cand_id, a.final, a.eval, u.name, u.email FROM apps a "
             "JOIN jobs j ON j.id=a.job_id JOIN users u ON u.id=a.cand_id "
             "WHERE a.job_id=? AND j.hr_id=? AND a.status='confirmed' AND a.final IS NOT NULL "
             "ORDER BY a.final DESC, a.id ASC", (job_id, hr_id))
    for r in rows:
        r["eval"] = json.loads(r["eval"] or "{}")
    return rows


# ---------- notifications ----------
def notify(user_id, job_id, text):
    try:
        ex("INSERT INTO notifs(user_id,job_id,text,ts) VALUES(?,?,?,?)", (user_id, job_id, text, time.time()))
        return True
    except sqlite3.IntegrityError:  # already notified for this job
        return False


def notifs(user_id):
    return q("SELECT * FROM notifs WHERE user_id=? ORDER BY id DESC", (user_id,))


def unseen(user_id):
    return q("SELECT COUNT(*) n FROM notifs WHERE user_id=? AND seen=0", (user_id,), one=True)["n"]


def mark_seen(user_id):
    ex("UPDATE notifs SET seen=1 WHERE user_id=?", (user_id,))


# ---------- prefs / memory (kept tiny on purpose: token cost) ----------
def pref(uid, k, default=""):
    r = q("SELECT v FROM prefs WHERE user_id=? AND k=?", (uid, k), one=True)
    return r["v"] if r else default


def set_pref(uid, k, v):
    ex("INSERT INTO prefs(user_id,k,v) VALUES(?,?,?) ON CONFLICT(user_id,k) DO UPDATE SET v=excluded.v",
       (uid, k, str(v)[:400]))


def get_notes(uid):
    return [r["note"] for r in q("SELECT note FROM notes WHERE user_id=? ORDER BY id", (uid,))]


def add_note(uid, note):
    note = (note or "").strip()[:120]
    if len(note) < 4:
        return
    if note.lower() in [n.lower() for n in get_notes(uid)]:
        return
    ex("INSERT INTO notes(user_id,note) VALUES(?,?)", (uid, note))
    ex("DELETE FROM notes WHERE user_id=? AND id NOT IN "
       "(SELECT id FROM notes WHERE user_id=? ORDER BY id DESC LIMIT 12)", (uid, uid))


def add_msg(uid, scope, role, content):
    ex("INSERT INTO msgs(user_id,scope,role,content) VALUES(?,?,?,?)", (uid, scope, role, content[:700]))


def history(uid, scope, limit=8):
    rows = q("SELECT role, content FROM msgs WHERE user_id=? AND scope=? ORDER BY id DESC LIMIT ?",
             (uid, scope, limit))
    return list(reversed(rows))


def all_msgs(uid, scope):
    return q("SELECT role, content FROM msgs WHERE user_id=? AND scope=? ORDER BY id", (uid, scope))


def trim_msgs(uid, scope, keep):
    ex("DELETE FROM msgs WHERE user_id=? AND scope=? AND id NOT IN "
       "(SELECT id FROM msgs WHERE user_id=? AND scope=? ORDER BY id DESC LIMIT ?)",
       (uid, scope, uid, scope, keep))


def clear_msgs(uid, scope):
    ex("DELETE FROM msgs WHERE user_id=? AND scope=?", (uid, scope))
