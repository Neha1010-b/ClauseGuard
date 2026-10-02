/* Upload page logic */

let selectedFile = null;

async function initUpload() {
    const user = await bootAuthedPage("upload");
    if (!user) return;

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("border-emerald-500", "bg-emerald-50/30");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-emerald-500", "bg-emerald-50/30");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-emerald-500", "bg-emerald-50/30");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (!file) return;
    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast("File exceeds 20 MB limit", "error");
        return;
    }
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["pdf", "docx", "txt"].includes(ext)) {
        showToast("Unsupported format. Use PDF, DOCX, or TXT.", "error");
        return;
    }
    selectedFile = file;
    // Start analysis immediately
    analyzeFile(file);
}

async function analyzeFile(file) {
    document.getElementById("upload-view").classList.add("hidden");
    document.getElementById("error-view").classList.add("hidden");
    document.getElementById("analyzing-view").classList.remove("hidden");

    const statusEl = document.getElementById("analyzing-status");
    const statuses = [
        "Extracting text from document",
        "Splitting into clauses",
        "Classifying clause types",
        "Comparing against reference bank",
        "Scoring risk",
        "Generating AI explanations",
    ];
    let idx = 0;
    const statusInterval = setInterval(() => {
        idx = (idx + 1) % statuses.length;
        statusEl.textContent = statuses[idx];
    }, 4000);

    const skipLlm = document.getElementById("skip-llm").checked;

    try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("generate_explanations", skipLlm ? "false" : "true");

        const endpoint = skipLlm ? "/analyze/quick" : "/analyze";
        const result = await apiFetch(endpoint, { method: "POST", body: formData });

        clearInterval(statusInterval);
        statusEl.textContent = "Saving results...";

        // Persist to the user's document history
        let savedId;
        try {
            const saved = await apiFetch("/documents", {
                method: "POST",
                body: {
                    filename: file.name,
                    analysis: result,
                },
            });
            savedId = saved.id;
        } catch (e) {
            // If save fails, still show results — just with a warning
            console.warn("Save to history failed:", e);
            sessionStorage.setItem("last_analysis", JSON.stringify(result));
            window.location.href = "/ui/results.html?temp=1";
            return;
        }

        window.location.href = `/ui/results.html?id=${encodeURIComponent(savedId)}`;
    } catch (err) {
        clearInterval(statusInterval);
        document.getElementById("analyzing-view").classList.add("hidden");
        document.getElementById("error-view").classList.remove("hidden");
        document.getElementById("error-message").textContent = err.message || "Analysis failed";
    }
}

initUpload();