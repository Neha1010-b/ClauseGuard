/* Dashboard page logic */

async function loadDashboard() {
    const user = await bootAuthedPage("dashboard");
    if (!user) return;

    // Greeting
    const hour = new Date().getHours();
    const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
    document.getElementById("greeting").textContent = `${greeting}, ${user.full_name.split(" ")[0]}`;

    try {
        const docs = await apiFetch("/documents");
        renderStats(docs);
        renderRecentDocs(docs);
    } catch (e) {
        showToast("Failed to load documents: " + e.message, "error");
        document.getElementById("recent-docs").innerHTML = `
            <div class="p-8 text-center text-slate-500 text-sm">Could not load documents.</div>`;
    }
}

function renderStats(docs) {
    const totalDocs = docs.length;
    const totalHigh = docs.reduce((s, d) => s + (d.high_count || 0), 0);
    const totalMed = docs.reduce((s, d) => s + (d.medium_count || 0), 0);
    const totalClauses = docs.reduce((s, d) => s + (d.total_clauses || 0), 0);

    document.getElementById("stat-documents").textContent = totalDocs;
    document.getElementById("stat-high").textContent = totalHigh;
    document.getElementById("stat-medium").textContent = totalMed;
    document.getElementById("stat-clauses").textContent = totalClauses;
}

function renderRecentDocs(docs) {
    const container = document.getElementById("recent-docs");
    if (!docs.length) {
        container.innerHTML = `
            <div class="p-12 text-center">
                <div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4">
                    <svg class="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                </div>
                <h3 class="text-sm font-medium text-slate-900 mb-1">No documents yet</h3>
                <p class="text-sm text-slate-500 mb-4">Upload a contract to get started.</p>
                <a href="/ui/upload.html" class="inline-block bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors">
                    Upload a document
                </a>
            </div>
        `;
        return;
    }

    container.innerHTML = docs.slice(0, 10).map(d => {
        const level = d.high_count > 0 ? "high" : d.medium_count > 0 ? "medium" : "low";
        const levelColors = {
            high: "bg-red-50 text-red-700 border-red-200",
            medium: "bg-amber-50 text-amber-700 border-amber-200",
            low: "bg-emerald-50 text-emerald-700 border-emerald-200",
        };
        const levelLabels = { high: "High risk", medium: "Medium", low: "Low risk" };

        return `
            <a href="/ui/results.html?id=${encodeURIComponent(d.id)}"
               class="flex items-center justify-between px-6 py-4 hover:bg-slate-50 transition-colors">
                <div class="flex items-center gap-4 min-w-0">
                    <div class="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0">
                        <svg class="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                    </div>
                    <div class="min-w-0">
                        <div class="text-sm font-medium text-slate-900 truncate">${escapeHtml(d.filename)}</div>
                        <div class="text-xs text-slate-500">
                            ${d.pages} pages · ${d.total_clauses} clauses ·
                            ${d.high_count}H / ${d.medium_count}M / ${d.low_count}L
                        </div>
                    </div>
                </div>
                <div class="flex items-center gap-4 flex-shrink-0">
                    <span class="text-xs px-2.5 py-1 rounded-full border font-medium ${levelColors[level]}">
                        ${levelLabels[level]}
                    </span>
                    <span class="text-xs text-slate-400">${formatDate(d.created_at)}</span>
                    <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
                </div>
            </a>
        `;
    }).join("");
}

loadDashboard();