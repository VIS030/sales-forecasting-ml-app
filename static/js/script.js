const formatCurrency = (value) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0);

const safeJson = (id) => {
  const node = document.getElementById(id);
  if (!node) return null;
  try {
    return JSON.parse(node.textContent);
  } catch (err) {
    return null;
  }
};

const chartPalette = ["#2563eb", "#0ea5e9", "#8b5cf6", "#f97316", "#10b981", "#e11d48"];

function initHomeChart() {
  const preview = safeJson("home-preview-data");
  const canvas = document.getElementById("homeTrendChart");
  if (!canvas || !preview) return;

  new Chart(canvas, {
    type: "line",
    data: {
      labels: preview.map((d) => d.x),
      datasets: [
        {
          data: preview.map((d) => d.y),
          borderColor: "#bfdbfe",
          pointBackgroundColor: "#ffffff",
          pointRadius: 2,
          borderWidth: 2.5,
          fill: true,
          backgroundColor: "rgba(191, 219, 254, 0.2)",
          tension: 0.35,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "rgba(255,255,255,.8)", maxTicksLimit: 6 }, grid: { display: false } },
        y: { ticks: { color: "rgba(255,255,255,.8)" }, grid: { color: "rgba(255,255,255,.08)" } },
      },
    },
  });
}

function initDashboardCharts() {
  const data = safeJson("dashboard-data");
  if (!data) return;

  const kpis = data.kpis || {};
  const revenueNode = document.getElementById("kpiRevenue");
  const ordersNode = document.getElementById("kpiOrders");
  const usersNode = document.getElementById("kpiUsers");
  const ratingNode = document.getElementById("kpiRating");
  if (revenueNode) revenueNode.textContent = formatCurrency(kpis.total_revenue);
  if (ordersNode) ordersNode.textContent = (kpis.total_orders || 0).toLocaleString();
  if (usersNode) usersNode.textContent = (kpis.unique_users || 0).toLocaleString();
  if (ratingNode) ratingNode.textContent = (kpis.avg_rating || 0).toFixed(2);

  const common = {
    plugins: { legend: { labels: { color: "#1e293b" } } },
    scales: { x: { ticks: { color: "#475569" } }, y: { ticks: { color: "#475569" } } },
  };

  const lineNode = document.getElementById("lineChart");
  if (lineNode) {
    new Chart(lineNode, {
      type: "line",
      data: {
        labels: data.trend.map((d) => d.x),
        datasets: [
          {
            label: "Sales",
            data: data.trend.map((d) => d.y),
            borderColor: "#2563eb",
            backgroundColor: "rgba(37,99,235,.12)",
            fill: true,
            tension: 0.3,
          },
        ],
      },
      options: common,
    });
  }

  const barNode = document.getElementById("barChart");
  if (barNode) {
    new Chart(barNode, {
      type: "bar",
      data: {
        labels: data.top_products.map((d) => d.label),
        datasets: [{ label: "Sales", data: data.top_products.map((d) => d.value), backgroundColor: "#0ea5e9" }],
      },
      options: { ...common, plugins: { legend: { display: false } } },
    });
  }

  const pieNode = document.getElementById("pieChart");
  if (pieNode) {
    new Chart(pieNode, {
      type: "pie",
      data: {
        labels: data.category_distribution.map((d) => d.label),
        datasets: [{ data: data.category_distribution.map((d) => d.value), backgroundColor: chartPalette }],
      },
      options: { plugins: { legend: { position: "bottom" } } },
    });
  }

  const scatterNode = document.getElementById("scatterChart");
  if (scatterNode) {
    new Chart(scatterNode, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Orders",
            data: data.discount_scatter,
            borderColor: "#f97316",
            backgroundColor: "rgba(249,115,22,.45)",
          },
        ],
      },
      options: {
        ...common,
        scales: {
          x: { title: { display: true, text: "DiscountRate" }, ticks: { color: "#475569" } },
          y: { title: { display: true, text: "TotalAmount" }, ticks: { color: "#475569" } },
        },
      },
    });
  }
}

function initPredictionForm() {
  const form = document.getElementById("predictForm");
  if (!form) return;

  const output = document.getElementById("predictionOutput");
  const spinner = document.getElementById("predictSpinner");
  const buttonText = form.querySelector(".btn-text");
  const miniChartCanvas = document.getElementById("predictionMiniChart");
  let predictionMiniChart;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = {
      category: formData.get("category"),
      country: formData.get("country"),
      price: Number(formData.get("price")),
      quantity: Number(formData.get("quantity")),
      discount_rate: Number(formData.get("discount_rate")),
    };

    spinner.classList.remove("d-none");
    buttonText.classList.add("d-none");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      output.innerHTML = `
        <div class="prediction-value">${formatCurrency(data.predicted_sales)}</div>
        <p class="prediction-meta mb-1">Confidence: ${data.confidence_score}%</p>
        <p class="text-soft mb-0">Model: ${data.model} | R2: ${data.evaluated_r2}</p>
      `;

      if (miniChartCanvas) {
        if (predictionMiniChart) predictionMiniChart.destroy();
        predictionMiniChart = new Chart(miniChartCanvas, {
          type: "bar",
          data: {
            labels: ["Predicted Sales", "Price x Qty"],
            datasets: [
              {
                data: [data.predicted_sales, payload.price * payload.quantity],
                backgroundColor: ["#2563eb", "#93c5fd"],
                borderRadius: 10,
              },
            ],
          },
          options: {
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { color: "#475569" } },
              y: { ticks: { color: "#475569" } },
            },
          },
        });
      }
    } catch (error) {
      output.innerHTML = `<p class="text-danger fw-semibold mb-0">${error.message}</p>`;
    } finally {
      spinner.classList.add("d-none");
      buttonText.classList.remove("d-none");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initHomeChart();
  initDashboardCharts();
  initPredictionForm();
});
