document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
  
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());


    const btn = e.target.querySelector('button');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        document.getElementById('result').classList.remove('hidden');
        const resultText = document.getElementById('prediction-text');
        
        resultText.innerText = result.prediction_text;
        
        
        document.getElementById('probability-text').innerText = result.probability;

    } catch (error) {
        console.error('Error:', error);
        alert("An error!");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});