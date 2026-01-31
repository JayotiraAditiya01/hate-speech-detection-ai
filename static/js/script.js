document.addEventListener("DOMContentLoaded", () => {

    const form = document.querySelector("form");
    const textarea = document.querySelector("textarea");
    const button = document.querySelector("button");
    const loader = document.getElementById("loader");
    const resultBox = document.getElementById("result-box");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const text = textarea.value.trim();
        if (!text) return;

        /* =============================
           RESET UI STATE
        ============================= */
        button.disabled = true;
        loader.style.display = "block";
        resultBox.style.display = "none";
        resultBox.innerHTML = "";

        try {
            /* =============================
               SEND REQUEST TO BACKEND
            ============================= */
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            /* =============================
               PROCESS RESPONSE
            ============================= */
            const isHate = data.prediction.toLowerCase().includes("hate");
            const confidence = data.confidence;

            resultBox.className = `result ${isHate ? "hate" : "safe"}`;
            resultBox.innerHTML = `
                ${isHate ? "🚨" : "✅"} ${data.prediction}
                <div class="confidence">
                    Confidence: <span>${confidence}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${isHate ? "hate-bar" : ""}"></div>
                    </div>
                </div>
            `;

            resultBox.style.display = "block";

            /* =============================
               ANIMATE CONFIDENCE BAR
            ============================= */
            setTimeout(() => {
                const fill = resultBox.querySelector(".confidence-fill");
                fill.style.width = confidence + "%";
            }, 100);

        } catch (error) {
            /* =============================
               ERROR HANDLING
            ============================= */
            resultBox.className = "result hate";
            resultBox.textContent = "Something went wrong. Please try again.";
            resultBox.style.display = "block";
        }

        /* =============================
           RESET LOADER & BUTTON
        ============================= */
        loader.style.display = "none";
        button.disabled = false;
    });

});
