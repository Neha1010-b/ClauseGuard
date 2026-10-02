/* ============================================================
   Contract Clause Risk Analyzer — Shared Frontend Library
   Handles: auth state, API calls, common UI helpers.
   ============================================================ */

const API_BASE = "/";       // same origin as backend

// ------------------------------------------------------------
// API client
// ------------------------------------------------------------
async function apiFetch(path, options = {}) {
    const url = path.startsWith("http") ? path : API_BASE + path.replace(/^\//, "");
    const opts = {
        credentials: "include",   // send/receive cookies
        ...options,
        headers: {
            "Accept": "application/json",
            ...(options.headers || {}),
        },
    };
    // If body is a plain object, JSON-encode it
    if (opts.body && typeof opts.body === "object" && !(opts.body instanceof FormData)) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(opts.body);
    }

    const resp = await fetch(url, opts);
    let data = null;
    const text = await resp.text();
    if (text) {
        try { data = JSON.parse(text); } catch { data = text; }
    }
    if (!resp.ok) {
        const err = new Error(
            (data && data.detail) || (data && data.error) || `HTTP ${resp.status}`
        );
        err.status = resp.status;
        err.data = data;
        throw err;
    }
    return data;
}

// ------------------------------------------------------------
// Auth state
// ------------------------------------------------------------
async function getCurrentUser() {
    try {
        return await apiFetch("/auth/me");
    } catch (e) {
        if (e.status === 401) return null;
        throw e;
    }
}

async function requireAuth() {
    const user = await getCurrentUser();
    if (!user) {
        window.location.href = "/ui/signin.html?next=" + encodeURIComponent(window.location.pathname);
        return null;
    }
    return user;
}

async function redirectIfAuthed() {
    const user = await getCurrentUser();
    if (user) {
        window.location.href = "/ui/dashboard.html";
        return user;
    }
    return null;
}

async function signout() {
    try { await apiFetch("/auth/signout", { method: "POST" }); }
    catch (e) { console.warn("Signout error:", e); }
    window.location.href = "/ui/index.html";
}

// ------------------------------------------------------------
// UI helpers
// ------------------------------------------------------------
function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function formatDate(iso) {
    if (!iso) return "—";
    try {
        const d = new Date(iso + (iso.endsWith("Z") ? "" : "Z"));
        return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    } catch { return iso; }
}

function formatTime(iso) {
    if (!iso) return "—";
    try {
        const d = new Date(iso + (iso.endsWith("Z") ? "" : "Z"));
        return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    } catch { return ""; }
}

function initialsFromName(name) {
    if (!name) return "?";
    const parts = name.trim().split(/\s+/);
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function showToast(message, type = "info") {
    const colors = {
        info: "bg-slate-800 text-white",
        success: "bg-emerald-600 text-white",
        error: "bg-red-600 text-white",
        warning: "bg-amber-500 text-white",
    };
    const toast = document.createElement("div");
    toast.className = `fixed bottom-6 right-6 z-50 px-4 py-3 rounded-lg shadow-lg ${colors[type] || colors.info} text-sm font-medium max-w-sm animate-slide-in`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ------------------------------------------------------------
// Page layout — sidebar user block
// ------------------------------------------------------------
function renderUserBlock(user) {
    const el = document.getElementById("user-block");
    if (!el) return;
    const initials = initialsFromName(user.full_name);
    el.innerHTML = `
        <div class="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-slate-700/50 transition-colors cursor-pointer" id="user-menu-btn">
            <div class="w-9 h-9 rounded-full bg-emerald-500 flex items-center justify-center text-white font-semibold text-sm">
                ${escapeHtml(initials)}
            </div>
            <div class="flex-1 min-w-0">
                <div class="text-sm font-medium text-white truncate">${escapeHtml(user.full_name)}</div>
                <div class="text-xs text-slate-400 truncate">${escapeHtml(user.email)}</div>
            </div>
        </div>
        <button onclick="signout()" class="w-full mt-2 text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded text-left transition-colors">
            Sign out
        </button>
    `;
}

// ------------------------------------------------------------
// Sidebar nav — highlighting current page
// ------------------------------------------------------------
function renderSidebarNav(activeKey) {
    const items = [
        { key: "dashboard", href: "/ui/dashboard.html", label: "Dashboard", icon: "grid" },
        { key: "upload",    href: "/ui/upload.html",    label: "Upload",    icon: "upload" },
        { key: "results",   href: "#",                  label: "Analyze",   icon: "chart" },
    ];
    const el = document.getElementById("nav");
    if (!el) return;

    const icons = {
        grid: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>',
        upload: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>',
        chart:  '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>',
    };

    el.innerHTML = items.map(it => {
        const active = it.key === activeKey;
        const cls = active
            ? "bg-slate-800 text-white"
            : "text-slate-300 hover:bg-slate-800/60 hover:text-white";
        return `
            <a href="${it.href}" class="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${cls}">
                ${icons[it.icon] || ""}
                <span class="text-sm font-medium">${it.label}</span>
            </a>
        `;
    }).join("");
}

// ------------------------------------------------------------
// Boot — called at the end of each authed page
// ------------------------------------------------------------
async function bootAuthedPage(activeKey) {
    const user = await requireAuth();
    if (!user) return null;
    renderUserBlock(user);
    renderSidebarNav(activeKey);
    return user;
}

// Expose helpers globally (no modules — simpler)
window.apiFetch = apiFetch;
window.getCurrentUser = getCurrentUser;
window.requireAuth = requireAuth;
window.redirectIfAuthed = redirectIfAuthed;
window.signout = signout;
window.escapeHtml = escapeHtml;
window.formatDate = formatDate;
window.formatTime = formatTime;
window.initialsFromName = initialsFromName;
window.showToast = showToast;
window.renderUserBlock = renderUserBlock;
window.renderSidebarNav = renderSidebarNav;
window.bootAuthedPage = bootAuthedPage;