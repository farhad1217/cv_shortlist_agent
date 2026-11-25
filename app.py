"""CV Review & Shortlisting Agent - single-file Streamlit UI.  Run:  streamlit run app.py"""
import hashlib
import json
import os

import streamlit as st

import db
import guards

st.set_page_config(page_title="CV Review & Shortlisting Agent", page_icon="🧭", layout="wide")

CSS = """
<style>
.block-container{padding-top:2rem;max-width:1300px}
html,body,[class*="css"]{font-family:"Segoe UI",system-ui,sans-serif}
h1,h2,h3{font-weight:600;color:#e7e7e7}
.hero{background:#252526;border:1px solid #3c3c3c;border-left:4px solid #007acc;
      border-radius:6px;padding:14px 20px;margin-bottom:16px}
.hero h2{margin:0;color:#e7e7e7;font-size:1.35rem}.hero p{margin:4px 0 0;color:#9d9d9d}
.chip{display:inline-block;background:#37373d;color:#9cdcfe;border-radius:4px;padding:1px 9px;
      margin:2px 4px 2px 0;font-size:.78rem;border:1px solid #4a4a50;font-family:Consolas,"Courier New",monospace}
.chip.bad{color:#f48771;border-color:#6b3b34}
.chip.good{color:#4ec9b0;border-color:#2f5f55}
.score{font-size:2rem;font-weight:700;color:#4fc1ff;font-family:Consolas,"Courier New",monospace}
[data-testid="stSidebar"]{background:#252526;border-right:1px solid #333}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:6px;border-color:#3c3c3c}
input,textarea,[data-baseweb="select"]>div{background:#3c3c3c!important;border-radius:3px!important}
.stButton>button{border-radius:3px;border:1px solid #4a4a50}
.stButton>button[kind="primary"]{background:#0e639c;border-color:#0e639c;color:#fff}
.stButton>button[kind="primary"]:hover{background:#1177bb;border-color:#1177bb}
[data-testid="stChatMessage"]{background:#252526;border:1px solid #333;border-radius:6px}
[data-testid="stExpander"]{border:1px solid #3c3c3c;border-radius:6px;background:#252526}
.tag{color:#c586c0;font-family:Consolas,monospace;font-size:.8rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting up (first run downloads a small embedding model)...")
def boot():
    db.init()
    import agent  # noqa: F401  (imports LangChain/LangGraph)
    import rag
    rag.ensure_kb()
    return True


try:
    boot()
    import agent
except Exception as e:  # noqa
    st.error(f"Startup failed: {e}")
    st.info("Did you run `pip install -r requirements.txt` inside your virtual environment?")
    st.stop()


def chips(items, kind=""):
    return "".join(f'<span class="chip {kind}">{i}</span>' for i in items) or "<span class='chip'>-</span>"


def extras_chips(j):
    return " ".join(f'<span class="chip">{e["name"]}: {e["value"]}</span>' for e in j.get("extras", []))


def hero(title, sub):
    st.markdown(f'<div class="hero"><h2>{title}</h2><p>{sub}</p></div>', unsafe_allow_html=True)


# =====================================================================
# Auth
# =====================================================================
def auth_page():
    hero("🧭 CV Review & Shortlisting Agent", "AI-assisted hiring: post jobs, apply with a PDF CV, get an explainable ranking.")
    t1, t2 = st.tabs(["Login", "Create account"])
    with t1:
        with st.form("login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                u = db.login(email, pw)
                if u:
                    st.session_state.user = u
                    st.rerun()
                st.error("Wrong email or password.")
    with t2:
        role = st.radio("I am a", ["Candidate", "HR"], horizontal=True)
        with st.form("register"):
            name = st.text_input("Full name" if role == "Candidate" else "Name / Company")
            email = st.text_input("Email" + (" (must end with @hr.com)" if role == "HR" else ""))
            pw = st.text_input("Password (min 6 chars)", type="password")
            if st.form_submit_button("Create account", type="primary"):
                u, err = db.register(email, name, pw, role.lower() if role == "HR" else "candidate")
                if err:
                    st.error(err)
                else:
                    st.session_state.user = u
                    st.rerun()


# =====================================================================
# Shared: chat panel (Cursor-style suggestions)
# =====================================================================
def chat_panel(user, scope, extra="", placeholder="Ask something...", on_draft=None, quick=None):
    uid = user["id"]
    nosug = db.pref(uid, "nosug") == "1"
    sug_key, err_key = f"sug:{scope}", f"err:{scope}"
    for m in db.all_msgs(uid, scope)[-14:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if st.session_state.get(err_key):
        st.warning(st.session_state.pop(err_key))

    inject = st.session_state.pop(f"inject:{scope}", None)
    if quick:
        qc = st.columns(len(quick))
        for i, (label, text) in enumerate(quick):
            if qc[i].button(label, key=f"q:{scope}:{i}", use_container_width=True):
                inject = text
    sugs = [] if nosug else st.session_state.get(sug_key, [])
    if sugs:
        st.caption("Suggested replies - click one or type your own:")
        cols = st.columns(len(sugs) + 1)
        for i, s in enumerate(sugs):
            if cols[i].button(s, key=f"{sug_key}{i}", use_container_width=True):
                inject = s
        if cols[-1].button("🚫 No suggestions", key=f"{sug_key}x", use_container_width=True):
            db.set_pref(uid, "nosug", "1")
            st.session_state[sug_key] = []
            st.rerun()
    elif nosug:
        if st.button("Turn AI suggestions back on", key=f"{sug_key}on"):
            db.set_pref(uid, "nosug", "0")
            st.rerun()

    c1, _ = st.columns([1, 4])
    if c1.button("🧹 Clear chat", key=f"clr{scope}"):
        db.clear_msgs(uid, scope)
        db.set_pref(uid, f"sum:{scope}", "")
        st.session_state[sug_key] = []
        st.rerun()

    prompt = st.chat_input(placeholder) or inject
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            res = agent.chat(user, scope, prompt, extra)
        if not res["ok"]:
            st.session_state[err_key] = res["error"]
        else:
            st.session_state[sug_key] = res["suggestions"]
            if res.get("draft") and on_draft:
                on_draft(res["draft"])
        st.rerun()


# =====================================================================
# HR pages
# =====================================================================
def _draft_sync(v):
    """Read what HR currently typed in the widgets back into the draft (used by the buttons' callbacks)."""
    d = st.session_state.draft
    d["title"] = st.session_state.get(f"t{v}", d["title"])
    d["description"] = st.session_state.get(f"d{v}", d["description"])
    d["skills"] = [x.strip() for x in st.session_state.get(f"s{v}", ", ".join(d["skills"])).split(",") if x.strip()]
    d["min_years"] = st.session_state.get(f"y{v}", d["min_years"])
    d["extras"] = [{"name": e["name"], "value": st.session_state.get(f"x{v}_{i}", e["value"])}
                   for i, e in enumerate(d["extras"])]


def _add_field(v):
    name = st.session_state.get("newf", "").strip()[:40]
    if not name or guards.injection_hit(name):
        return
    _draft_sync(v)
    d = st.session_state.draft
    if name.lower() not in [e["name"].lower() for e in d["extras"]] and len(d["extras"]) < 10:
        d["extras"].append({"name": name, "value": ""})
        st.session_state.dv = v + 1
    st.session_state["newf"] = ""


def _del_field(v, i):
    _draft_sync(v)
    st.session_state.draft["extras"].pop(i)
    st.session_state.dv = v + 1


def hr_post(user):
    hero("Post a Job", "Draft on the left, AI assistant on the right. Clicking a suggestion applies it to the draft.")
    d = st.session_state.setdefault("draft", {"title": "", "description": "", "skills": [], "min_years": 0.0,
                                             "extras": []})
    d.setdefault("extras", [])
    v = st.session_state.get("dv", 0)
    left, right = st.columns([5, 6], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### 📝 Job draft")
            title = st.text_input("Job title", d["title"], key=f"t{v}")
            desc = st.text_area("Description & responsibilities", d["description"], height=170, key=f"d{v}")
            skills_s = st.text_input("Required skills (comma separated)", ", ".join(d["skills"]), key=f"s{v}")
            yrs = st.number_input("Minimum years of experience", 0.0, 40.0, float(d["min_years"] or 0), 0.5, key=f"y{v}")
            skills = [x.strip() for x in skills_s.split(",") if x.strip()][:12]
            extras = []
            for i, e in enumerate(d["extras"]):
                c1, c2 = st.columns([6, 1])
                val = c1.text_input(e["name"], e["value"], key=f"x{v}_{i}")
                c2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
                c2.button("✕", key=f"rm{v}_{i}", on_click=_del_field, args=(v, i), help="Remove this field")
                extras.append({"name": e["name"], "value": val})
            c1, c2 = st.columns([5, 2])
            c1.text_input("Add your own field", key="newf", placeholder="e.g. Salary, Location, Work mode",
                          label_visibility="collapsed")
            c2.button("➕ Add field", on_click=_add_field, args=(v,), use_container_width=True)
            st.divider()
            st.caption("Nothing goes live until you press Publish (your final approval).")
            if st.button("✅ Publish job", type="primary", use_container_width=True):
                clean = [e for e in extras if e["name"].strip() and e["value"].strip()]
                alltext = title + " " + desc + " " + " ".join(e["name"] + " " + e["value"] for e in clean)
                if len(title.strip()) < 3 or len(desc.strip()) < 40 or not skills:
                    st.error("Please provide a title, a description (40+ chars) and at least one skill.")
                elif guards.cv_injection_hit(alltext):
                    st.error("The text contains instructions aimed at AI systems. Please remove them.")
                else:
                    agent.publish_job(user["id"], title.strip()[:120], desc.strip()[:3000], skills, yrs, clean)
                    st.session_state.draft = {"title": "", "description": "", "skills": [], "min_years": 0.0,
                                              "extras": []}
                    st.session_state.dv = v + 1
                    db.clear_msgs(user["id"], "hr_post")
                    st.session_state.pop("sug:hr_post", None)
                    st.success("Job published! Find it under 'My Jobs & Applicants'.")

    def on_draft(nd):
        cur = st.session_state.draft
        st.session_state.draft = {
            "title": nd.get("title") or cur["title"], "description": nd.get("description") or cur["description"],
            "skills": nd.get("skills") or cur["skills"], "min_years": nd.get("min_years") or cur["min_years"],
            "extras": nd.get("extras") or cur["extras"]}
        st.session_state.dv = st.session_state.get("dv", 0) + 1

    extra = "Current draft (as edited by HR): " + json.dumps(
        {"title": title, "description": desc[:700], "skills": skills, "min_years": yrs,
         "extras": extras}, ensure_ascii=False)
    with right:
        st.markdown("#### 🤖 AI job assistant")
        chat_panel(user, "hr_post", extra, "e.g. I need a backend engineer for a fintech team...", on_draft)


def start_edit(job):
    st.session_state["edit"] = {"id": job["id"], "v": 0, "d": {
        "title": job["title"], "description": job["description"], "skills": list(job["skills"]),
        "min_years": float(job["min_years"] or 0), "extras": [dict(e) for e in job["extras"]]}}


def _cancel_edit():
    st.session_state.pop("edit", None)


def _edit_sync():
    ed = st.session_state["edit"]
    d, k = ed["d"], f"{ed['id']}_{ed['v']}"
    d["title"] = st.session_state.get(f"et{k}", d["title"])
    d["description"] = st.session_state.get(f"ed{k}", d["description"])
    d["skills"] = [x.strip() for x in st.session_state.get(f"es{k}", ", ".join(d["skills"])).split(",") if x.strip()]
    d["min_years"] = st.session_state.get(f"ey{k}", d["min_years"])
    d["extras"] = [{"name": e["name"], "value": st.session_state.get(f"ex{k}_{i}", e["value"])}
                   for i, e in enumerate(d["extras"])]


def _edit_add_field():
    name = st.session_state.get("enewf", "").strip()[:40]
    if name and not guards.injection_hit(name):
        _edit_sync()
        ed = st.session_state["edit"]
        if name.lower() not in [e["name"].lower() for e in ed["d"]["extras"]] and len(ed["d"]["extras"]) < 10:
            ed["d"]["extras"].append({"name": name, "value": ""})
            ed["v"] += 1
    st.session_state["enewf"] = ""


def _edit_del_field(i):
    _edit_sync()
    ed = st.session_state["edit"]
    ed["d"]["extras"].pop(i)
    ed["v"] += 1


def edit_form(user, job):
    ed = st.session_state["edit"]
    d, k = ed["d"], f"{ed['id']}_{ed['v']}"
    with st.container(border=True):
        st.markdown(f"#### ✏️ Edit job #{job['id']}")
        title = st.text_input("Job title", d["title"], key=f"et{k}")
        desc = st.text_area("Description & responsibilities", d["description"], height=180, key=f"ed{k}")
        skills_s = st.text_input("Required skills (comma separated)", ", ".join(d["skills"]), key=f"es{k}")
        yrs = st.number_input("Minimum years of experience", 0.0, 40.0, float(d["min_years"] or 0), 0.5, key=f"ey{k}")
        skills = [x.strip() for x in skills_s.split(",") if x.strip()][:12]
        extras = []
        for i, e in enumerate(d["extras"]):
            c1, c2 = st.columns([6, 1])
            val = c1.text_input(e["name"], e["value"], key=f"ex{k}_{i}")
            c2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            c2.button("✕", key=f"erm{k}_{i}", on_click=_edit_del_field, args=(i,), help="Remove this field")
            extras.append({"name": e["name"], "value": val})
        c1, c2 = st.columns([5, 2])
        c1.text_input("Add your own field", key="enewf", placeholder="e.g. Salary, Location, Work mode",
                      label_visibility="collapsed")
        c2.button("➕ Add field", on_click=_edit_add_field, use_container_width=True)
        if job.get("status") == "open" and db.job_applicants(job["id"], user["id"]):
            st.caption("Note: existing applicants keep the scores they already received.")
        b1, b2 = st.columns(2)
        b2.button("Cancel", on_click=_cancel_edit, use_container_width=True)
        if b1.button("💾 Save changes", type="primary", use_container_width=True):
            clean = [e for e in extras if e["name"].strip() and e["value"].strip()]
            alltext = title + " " + desc + " " + " ".join(e["name"] + " " + e["value"] for e in clean)
            if len(title.strip()) < 3 or len(desc.strip()) < 40 or not skills:
                st.error("Please provide a title, a description (40+ chars) and at least one skill.")
            elif guards.cv_injection_hit(alltext):
                st.error("The text contains instructions aimed at AI systems. Please remove them.")
            else:
                agent.edit_job(job["id"], user["id"], title.strip()[:120], desc.strip()[:3000], skills, yrs, clean)
                st.session_state.pop("edit", None)
                st.session_state["flash"] = "Job updated."
                st.rerun()


def hr_jobs(user):
    hero("My Jobs & Applicants", "Ranked applicants, explainable scores and the approval-gated shortlist.")
    jobs = db.hr_jobs(user["id"])
    if not jobs:
        st.info("You have not posted any jobs yet.")
        return
    jid = st.selectbox("Select a job", [j["id"] for j in jobs],
                       format_func=lambda i: next(f"#{j['id']} {j['title']}  [{j['status']}]" for j in jobs if j["id"] == i))
    job = db.hr_job(jid, user["id"])
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    ed = st.session_state.get("edit")
    if ed and ed["id"] != jid:
        st.session_state.pop("edit", None)
        ed = None
    if ed:
        edit_form(user, job)
    else:
        with st.container(border=True):
            st.markdown(f"**{job['title']}**  ·  min {job['min_years']} yrs  ·  {job['status']}")
            st.markdown(chips(job["skills"]) + extras_chips(job), unsafe_allow_html=True)
            st.write(job["description"])
            b1, b2, _ = st.columns([1, 2, 3])
            b1.button("✏️ Edit job", on_click=start_edit, args=(job,), key=f"edit{jid}")
            if job["status"] == "open" and b2.button("Close job (stop new applications)", key=f"close{jid}"):
                agent.close_job(jid, user["id"])
                st.rerun()

    apps = db.job_applicants(jid, user["id"])
    st.subheader(f"Applicants ({len(apps)} submitted)")
    if not apps:
        st.caption("No submitted applications yet.")
    for i, a in enumerate(apps, 1):
        e = a["eval"]
        with st.expander(f"#{i}  {a['name']}  -  score {a['final']}"):
            c = st.columns(4)
            c[0].markdown(f'<div class="score">{a["final"]}</div>final score', unsafe_allow_html=True)
            c[1].metric("Skill match (50%)", e.get("skill"))
            c[2].metric("Projects (25%)", e.get("project"))
            c[3].metric("Experience (25%)", e.get("exp"))
            st.markdown("**Why this score**")
            st.write(e.get("summary", ""))
            st.markdown("Matched: " + chips(e.get("matched", []), "good"), unsafe_allow_html=True)
            st.markdown("Missing: " + chips(e.get("missing", []), "bad"), unsafe_allow_html=True)
            for k, t in (e.get("cites") or {}).items():
                st.caption(f"[{k}] {t}")
            p = db.cv_path(jid, a["cand_id"])
            if os.path.exists(p):
                st.download_button("⬇ Download CV (PDF)", open(p, "rb").read(), f"{a['name']}_cv.pdf",
                                   "application/pdf", key=f"dl{a['id']}")

    if apps:
        st.subheader("Shortlist & interview invitations")
        pending = agent.pending_shortlist(jid, user["id"])
        c1, c2 = st.columns([1, 3])
        k = c1.number_input("Top N", 1, max(len(apps), 1), min(5, len(apps)), key=f"k{jid}")
        if c2.button("Propose shortlist (AI ranking)", key=f"prop{jid}"):
            agent.start_shortlist(jid, user["id"], int(k))
            st.rerun()
        if pending is not None:
            st.info("A shortlist proposal is waiting for your approval. Add or remove anyone before approving.")
            names = {a["cand_id"]: f"{a['name']} ({a['final']})" for a in apps}
            chosen = st.multiselect("Candidates to invite for interview", list(names), default=[x for x in pending if x in names],
                                    format_func=lambda x: names[x], key=f"ms{jid}")
            if st.button("✅ Approve & send notifications", type="primary", key=f"ap{jid}", disabled=not chosen):
                n = agent.approve_shortlist(jid, user["id"], chosen)
                st.success(f"Notifications sent to {n} candidate(s).")
                st.rerun()
        st.subheader("Ask AI about these applicants")
        chat_panel(user, f"hr_job:{jid}", f"Focus job_id={jid} ('{job['title']}').",
                   "e.g. Compare the top 3 candidates")


# =====================================================================
# Candidate pages
# =====================================================================
def go_apply(jid):
    st.session_state["page"] = "My Applications"
    st.session_state["apply_job"] = jid


def cand_browse(user):
    hero("Browse Jobs", "Jobs are ranked by your interests (semantic search over all open postings).")
    interests = st.text_input("Your interests / target role", db.pref(user["id"], "interests"),
                              placeholder="e.g. python backend, machine learning, remote")
    if interests != db.pref(user["id"], "interests"):
        if guards.injection_hit(interests) or len(interests) > 200:
            st.warning("Please enter a short, plain description of your interests.")
        else:
            db.set_pref(user["id"], "interests", interests)
    jobs = db.open_jobs()
    if interests.strip() and jobs:
        import rag
        order = {jid: i for i, (jid, _) in enumerate(rag.match_jobs(interests, 50))}
        jobs.sort(key=lambda j: order.get(j["id"], 999))
    if not jobs:
        st.info("No open jobs right now.")
    for j in jobs:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{j['title']}**  ·  {j['hr_name']}  ·  min {j['min_years']} yrs")
            c1.markdown(chips(j["skills"]) + extras_chips(j), unsafe_allow_html=True)
            c1.caption(j["description"][:260] + ("..." if len(j["description"]) > 260 else ""))
            c2.button("Apply", key=f"ap{j['id']}", on_click=go_apply, args=(j["id"],), type="primary")


def ask_ai(scope, text):
    """Button callback: send a prompt to the chat on the right."""
    st.session_state[f"inject:{scope}"] = text


def cand_apply(user):
    hero("My Applications", "Workspace on the left, AI assistant on the right. Confirmed CVs are locked.")
    uid = user["id"]
    mine = {a["job_id"] for a in db.cand_apps(uid)}
    jobs = [j for j in db.open_jobs()] + [j for j in (db.get_job(i) for i in mine) if j and j["status"] != "open"]
    if not jobs:
        st.info("No jobs available.")
        return
    ids = [j["id"] for j in jobs]
    default = st.session_state.get("apply_job")
    jid = st.selectbox("Job", ids, index=ids.index(default) if default in ids else 0,
                       format_func=lambda i: next(f"#{j['id']} {j['title']} - {j['hr_name']}" for j in jobs if j["id"] == i))
    job = db.get_job(jid)
    left, right = st.columns([6, 5], gap="large")
    with left:
        apply_workspace(user, job)
    with right:
        st.markdown("#### 🤖 AI assistant")
        chat_panel(user, f"cand_job:{jid}", f"Focus job_id={jid} ('{job['title']}').",
                   "Ask about this job or your CV...")


def apply_workspace(user, job):
    uid, jid = user["id"], job["id"]
    scope = f"cand_job:{jid}"
    with st.container(border=True):
        st.markdown(f"**{job['title']}** · {job['hr_name']} · min {job['min_years']} yrs")
        st.markdown(chips(job["skills"]) + extras_chips(job), unsafe_allow_html=True)
        st.caption(job["description"][:300] + ("..." if len(job["description"]) > 300 else ""))
        st.markdown("**AI helpers** (answers appear in the chat)")
        h1, h2, h3 = st.columns(3)
        h1.button("📄 Summarize job", key=f"h1{jid}", use_container_width=True, on_click=ask_ai, args=(
            scope, "Summarize this job: role, key skills, level, and what I should emphasise."))
        h2.button("🔎 CV missing anything?", key=f"h2{jid}", use_container_width=True, on_click=ask_ai, args=(
            scope, "Review my uploaded CV against this job. What is missing or weak?"))
        h3.button("➕ What to add?", key=f"h3{jid}", use_container_width=True, on_click=ask_ai, args=(
            scope, "What exactly should I add or improve in my CV for this job?"))
    app = db.get_app(jid, uid)

    if app and app["status"] == "confirmed":
        st.success("✅ Your application is submitted and locked. It cannot be edited. "
                   "You'll get a notification here if HR shortlists you.")
        return
    if job["status"] != "open":
        st.warning("This job is closed.")
        return

    st.subheader("1 · Upload your CV (PDF only)")
    up = st.file_uploader("PDF file, max 5 MB, up to 6 pages", type=["pdf"], key=f"up{jid}")
    if up and st.button("Upload / replace CV", key=f"ub{jid}"):
        text, err = guards.read_pdf(up.getvalue())
        if err:
            st.error(err)
        elif guards.cv_injection_hit(text):
            st.error("This CV contains instructions aimed at the AI screening system and was rejected.")
        else:
            if db.save_draft(jid, uid, hashlib.md5(text.encode()).hexdigest()):
                open(db.cv_path(jid, uid), "wb").write(up.getvalue())
                st.success("CV uploaded. Now check it with the AI (optional) and confirm when ready.")
                st.rerun()
    app = db.get_app(jid, uid)
    if not app:
        return
    st.caption("A CV is uploaded (draft). You may replace it as often as you like before confirming.")

    st.subheader("2 · Check with AI (optional)")

    def cv_text():
        with open(db.cv_path(jid, uid), "rb") as f:
            return guards.read_pdf(f.read())

    def run_eval():
        text, err = cv_text()
        if err:
            return None, err, True
        out = agent.evaluate_cv(job, text)
        if "result" in out:
            db.save_eval(app["id"], out["result"], out["result"]["final"])
            return out["result"], "", False
        if out.get("rejected"):  # garbage / injection CV -> remove it
            db.ex("DELETE FROM apps WHERE id=? AND status='draft'", (app["id"],))
            try:
                os.remove(db.cv_path(jid, uid))
            except OSError:
                pass
        return None, out["error"], out.get("rejected", False)

    if st.button("🔍 Check my CV", key=f"chk{jid}"):
        with st.spinner("AI is reviewing your CV..."):
            res, err, rej = run_eval()
        if err:
            st.error(("CV rejected: " if rej else "") + err)
            if rej:
                return
        else:
            st.rerun()
    app = db.get_app(jid, uid)
    if app and app["eval"]:
        e = json.loads(app["eval"])
        c = st.columns(4)
        c[0].markdown(f'<div class="score">{e["final"]}</div>estimated score', unsafe_allow_html=True)
        c[1].metric("Skills", e["skill"])
        c[2].metric("Projects", e["project"])
        c[3].metric("Experience", e["exp"])
        st.write(e["summary"])
        st.markdown("Matched: " + chips(e["matched"], "good"), unsafe_allow_html=True)
        st.markdown("Missing: " + chips(e["missing"], "bad"), unsafe_allow_html=True)
        if e["improvements"]:
            st.markdown("**Tips to improve your CV**")
            for t in e["improvements"]:
                st.markdown(f"- {t}")

    st.subheader("3 · Confirm & submit")
    st.warning("After confirming you can NOT edit or replace this CV.")
    ok = st.checkbox("I understand and want to submit this CV for this job", key=f"cf{jid}")
    if st.button("Confirm & Submit", type="primary", disabled=not ok, key=f"sub{jid}"):
        if not (app and app["eval"]):
            with st.spinner("Reviewing your CV before submission..."):
                res, err, rej = run_eval()
            if err:
                st.error(("CV rejected: " if rej else "") + err)
                return
        app = db.get_app(jid, uid)
        db.confirm_app(app["id"])
        st.success("Submitted!")
        st.rerun()




def cand_coach(user):
    hero("CV Coach", "Ask what your CV should contain, in general or for a specific job.")
    jobs = db.open_jobs()
    opts = [0] + [j["id"] for j in jobs]
    jid = st.selectbox("Focus", opts, format_func=lambda i: "General advice" if i == 0 else
                       next(f"#{j['id']} {j['title']}" for j in jobs if j["id"] == i))
    extra = f"Focus job_id={jid}." if jid else ""
    chat_panel(user, f"cand_job:{jid}", extra, "e.g. What should my CV include for this job?")


def cand_notifs(user):
    hero("Notifications", "Interview invitations from HR.")
    ns = db.notifs(user["id"])
    if not ns:
        st.info("No notifications yet.")
    for n in ns:
        with st.container(border=True):
            st.markdown(("🆕 " if not n["seen"] else "") + n["text"])
    db.mark_seen(user["id"])


# =====================================================================
# Router
# =====================================================================
def main():
    user = st.session_state.get("user")
    if not user:
        auth_page()
        return
    with st.sidebar:
        st.markdown(f"### 🧭 {user['name']}")
        st.caption(f"{user['role'].upper()} · {user['email']}")
        pages = (["Post a Job", "My Jobs & Applicants"] if user["role"] == "hr" else
                 ["Browse Jobs", "My Applications", "CV Coach", "Notifications"])
        if user["role"] == "candidate":
            n = db.unseen(user["id"])
            if n:
                st.info(f"🔔 {n} new notification(s)")
        page = st.radio("Menu", pages, key="page")
        st.divider()
        sug = st.toggle("AI suggestions", value=db.pref(user["id"], "nosug") != "1")
        if sug != (db.pref(user["id"], "nosug") != "1"):
            db.set_pref(user["id"], "nosug", "0" if sug else "1")
            st.rerun()
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
    {"Post a Job": hr_post, "My Jobs & Applicants": hr_jobs, "Browse Jobs": cand_browse,
     "My Applications": cand_apply, "CV Coach": cand_coach, "Notifications": cand_notifs}[page](user)


main()
