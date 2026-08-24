const modelSelect = document.getElementById("model");
const thresholdInput = document.getElementById("threshold");
const thresholdVal = document.getElementById("thresholdVal");
const results = document.getElementById("results");
const errorBox = document.getElementById("error");
const predictButton = document.getElementById("predictBtn");
const titleInput = document.getElementById("title");
const descriptionInput = document.getElementById("description");


// Update displayed threshold value
thresholdInput.addEventListener("input", () => {
    thresholdVal.textContent = thresholdInput.value;
});


// Load available models
async function loadModels() {
    try {
        const response = await fetch("/api/models");

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Failed to load models");
        }

        const data = await response.json();

        modelSelect.innerHTML = data.models
            .map((model) => `<option value="${model}">${model}</option>`)
            .join("");
    } catch (error) {
        errorBox.textContent = error.message;
    }
}


// Predict tags
async function predict() {
    errorBox.textContent = "";
    results.innerHTML = "Predicting...";

    const payload = {
        model: modelSelect.value,
        title: titleInput.value,
        description: descriptionInput.value,
        threshold: parseFloat(thresholdInput.value),
    };

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        if (data.predictions.length === 0) {
            results.innerHTML =
                "<p>No tags passed the threshold. Try lowering it.</p>";

            return;
        }

        results.innerHTML = data.predictions
            .map(
                (prediction) => `
                    <div class="tag-row">
                        <span>${prediction.tag}</span>
                        <span>
                            ${(prediction.probability * 100).toFixed(1)}%
                        </span>
                    </div>
                `
            )
            .join("");
    } catch (error) {
        results.innerHTML = "";
        errorBox.textContent = error.message;
    }
}


// Event listeners
predictButton.addEventListener("click", predict);


// Initialize
loadModels();