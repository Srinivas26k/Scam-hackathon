// Customer Churn Prediction System - Interactive JavaScript

// Application data
const appData = {
  teamMembers: [
    {"name": "Sri", "role": "ML Engineer", "expertise": "Deep Learning & Feature Engineering"},
    {"name": "Na", "role": "Data Scientist", "expertise": "Statistical Analysis & Model Optimization"},
    {"name": "Ka", "role": "Full-Stack Developer", "expertise": "Deployment & UI/UX Design"}
  ],
  modelMetrics: {
    f1Score: 0.847,
    aucScore: 0.923,
    accuracy: 0.942,
    precision: 0.856,
    recall: 0.839
  },
  featureImportance: [
    {"feature": "tenure", "importance": 0.25},
    {"feature": "MonthlyCharges", "importance": 0.18},
    {"feature": "TotalCharges", "importance": 0.15},
    {"feature": "Contract_Month-to-month", "importance": 0.12},
    {"feature": "PaymentMethod_Electronic check", "importance": 0.08},
    {"feature": "InternetService_Fiber optic", "importance": 0.07},
    {"feature": "avg_monthly_charges", "importance": 0.05},
    {"feature": "total_services", "importance": 0.04},
    {"feature": "charges_per_service", "importance": 0.03},
    {"feature": "is_monthly_contract", "importance": 0.03}
  ],
  sampleCustomers: [
    {
      customerID: "CUST-0001",
      gender: "Male",
      SeniorCitizen: 0,
      Partner: "Yes",
      Dependents: "No",
      tenure: 24,
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "Yes",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: 89.5,
      TotalCharges: "2148.0",
      churnProbability: 0.78,
      predictedChurn: 1
    },
    {
      customerID: "CUST-0002",
      gender: "Female",
      SeniorCitizen: 1,
      Partner: "No",
      Dependents: "No",
      tenure: 48,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      DeviceProtection: "Yes",
      TechSupport: "Yes",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Two year",
      PaperlessBilling: "No",
      PaymentMethod: "Credit card (automatic)",
      MonthlyCharges: 65.2,
      TotalCharges: "3129.6",
      churnProbability: 0.12,
      predictedChurn: 0
    }
  ],
  churnAnalytics: {
    byContract: [
      {"contract": "Month-to-month", "churnRate": 0.43},
      {"contract": "One year", "churnRate": 0.11},
      {"contract": "Two year", "churnRate": 0.03}
    ],
    byTenure: [
      {"tenureGroup": "0-12 months", "churnRate": 0.47},
      {"tenureGroup": "13-24 months", "churnRate": 0.35},
      {"tenureGroup": "25-48 months", "churnRate": 0.15},
      {"tenureGroup": "49+ months", "churnRate": 0.08}
    ],
    byInternetService: [
      {"service": "Fiber optic", "churnRate": 0.42},
      {"service": "DSL", "churnRate": 0.19},
      {"service": "No", "churnRate": 0.07}
    ]
  }
};

// Chart colors
const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'];

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
  initNavigation();
  initDemo();
  initCharts();
  populateSampleData();
});

// Navigation functionality
function initNavigation() {
  const navButtons = document.querySelectorAll('.nav__btn');
  const sections = document.querySelectorAll('.section');

  navButtons.forEach(button => {
    button.addEventListener('click', function() {
      const targetSection = this.getAttribute('data-section');
      
      // Update active button
      navButtons.forEach(btn => btn.classList.remove('active'));
      this.classList.add('active');
      
      // Show target section
      sections.forEach(section => {
        section.classList.remove('section--active');
        if (section.id === targetSection) {
          section.classList.add('section--active');
        }
      });

      // Initialize charts when sections become visible
      if (targetSection === 'insights') {
        setTimeout(() => initFeatureChart(), 100);
      } else if (targetSection === 'analytics') {
        setTimeout(() => {
          initContractChart();
          initTenureChart();
          initInternetChart();
        }, 100);
      }
    });
  });

  // Set home as active by default
  document.querySelector('.nav__btn[data-section="home"]').classList.add('active');
}

// Demo functionality
function initDemo() {
  const form = document.getElementById('churnForm');
  let gaugeChart = null;

  form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(form);
    const customerData = {};
    
    for (let [key, value] of formData.entries()) {
      customerData[key] = value;
    }
    
    // Simple prediction logic based on key factors
    const prediction = predictChurn(customerData);
    displayPrediction(prediction);
    updateGaugeChart(prediction.probability);
  });

  // Initialize gauge chart
  function initGaugeChart() {
    const ctx = document.getElementById('gaugeChart').getContext('2d');
    gaugeChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [0, 100],
          backgroundColor: ['#1FB8CD', '#f0f0f0'],
          borderWidth: 0,
          cutout: '70%'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            enabled: false
          }
        },
        rotation: -90,
        circumference: 180
      }
    });
  }

  function updateGaugeChart(probability) {
    if (!gaugeChart) {
      initGaugeChart();
    }
    
    const percentage = Math.round(probability * 100);
    const color = percentage > 70 ? '#DB4545' : percentage > 40 ? '#D2BA4C' : '#1FB8CD';
    
    gaugeChart.data.datasets[0].data = [percentage, 100 - percentage];
    gaugeChart.data.datasets[0].backgroundColor = [color, '#f0f0f0'];
    gaugeChart.update();
  }

  // Initialize gauge chart on page load
  setTimeout(initGaugeChart, 100);
}

// Prediction logic
function predictChurn(customerData) {
  let score = 0;
  
  // Tenure factor (higher tenure = lower churn risk)
  const tenure = parseInt(customerData.tenure) || 0;
  if (tenure < 12) score += 0.3;
  else if (tenure < 24) score += 0.15;
  else if (tenure < 48) score += 0.05;
  
  // Contract factor
  if (customerData.Contract === 'Month-to-month') score += 0.25;
  else if (customerData.Contract === 'One year') score += 0.05;
  
  // Monthly charges factor
  const monthlyCharges = parseFloat(customerData.MonthlyCharges) || 0;
  if (monthlyCharges > 80) score += 0.15;
  else if (monthlyCharges > 60) score += 0.08;
  
  // Internet service factor
  if (customerData.InternetService === 'Fiber optic') score += 0.1;
  
  // Payment method factor
  if (customerData.PaymentMethod === 'Electronic check') score += 0.12;
  
  // Senior citizen factor
  if (customerData.SeniorCitizen === '1') score += 0.08;
  
  // Partner/Dependents factor
  if (customerData.Partner === 'No' && customerData.Dependents === 'No') {
    score += 0.1;
  }
  
  // Normalize score to 0-1 range
  const probability = Math.min(Math.max(score, 0), 1);
  
  let riskLevel = 'Low';
  let recommendation = 'Customer shows low churn risk. Continue standard engagement.';
  
  if (probability > 0.7) {
    riskLevel = 'High';
    recommendation = 'High churn risk! Immediate intervention recommended: offer retention incentives, personalized support, or contract renegotiation.';
  } else if (probability > 0.4) {
    riskLevel = 'Medium';
    recommendation = 'Moderate churn risk. Consider proactive engagement: check satisfaction, offer service upgrades, or loyalty programs.';
  }
  
  return {
    probability: probability,
    riskLevel: riskLevel,
    recommendation: recommendation
  };
}

// Display prediction results
function displayPrediction(prediction) {
  const riskLevelElement = document.getElementById('riskLevel');
  const churnProbElement = document.getElementById('churnProb');
  const recommendationElement = document.getElementById('recommendation');
  
  riskLevelElement.textContent = prediction.riskLevel;
  riskLevelElement.className = 'risk-value ' + prediction.riskLevel.toLowerCase();
  
  churnProbElement.textContent = (prediction.probability * 100).toFixed(1) + '%';
  recommendationElement.textContent = prediction.recommendation;
}

// Initialize all charts
function initCharts() {
  // Charts will be initialized when their sections become visible
}

// Feature importance chart
function initFeatureChart() {
  const ctx = document.getElementById('featureChart');
  if (!ctx || ctx.chart) return; // Avoid recreating chart
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: appData.featureImportance.map(item => item.feature),
      datasets: [{
        label: 'Feature Importance',
        data: appData.featureImportance.map(item => item.importance),
        backgroundColor: chartColors[0],
        borderColor: chartColors[0],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 0.3,
          title: {
            display: true,
            text: 'Importance Score'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Features'
          }
        }
      }
    }
  });
}

// Contract type chart
function initContractChart() {
  const ctx = document.getElementById('contractChart');
  if (!ctx || ctx.chart) return;
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: appData.churnAnalytics.byContract.map(item => item.contract),
      datasets: [{
        label: 'Churn Rate',
        data: appData.churnAnalytics.byContract.map(item => item.churnRate * 100),
        backgroundColor: chartColors.slice(0, 3),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 50,
          title: {
            display: true,
            text: 'Churn Rate (%)'
          }
        }
      }
    }
  });
}

// Tenure chart
function initTenureChart() {
  const ctx = document.getElementById('tenureChart');
  if (!ctx || ctx.chart) return;
  
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: appData.churnAnalytics.byTenure.map(item => item.tenureGroup),
      datasets: [{
        label: 'Churn Rate',
        data: appData.churnAnalytics.byTenure.map(item => item.churnRate * 100),
        borderColor: chartColors[1],
        backgroundColor: chartColors[1] + '20',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 50,
          title: {
            display: true,
            text: 'Churn Rate (%)'
          }
        }
      }
    }
  });
}

// Internet service chart
function initInternetChart() {
  const ctx = document.getElementById('internetChart');
  if (!ctx || ctx.chart) return;
  
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: appData.churnAnalytics.byInternetService.map(item => item.service),
      datasets: [{
        data: appData.churnAnalytics.byInternetService.map(item => item.churnRate * 100),
        backgroundColor: chartColors.slice(2, 5),
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom'
        }
      }
    }
  });
}

// Populate sample data table
function populateSampleData() {
  const tbody = document.getElementById('sampleDataBody');
  if (!tbody) return;
  
  appData.sampleCustomers.forEach(customer => {
    const row = document.createElement('tr');
    
    const riskClass = customer.churnProbability > 0.7 ? 'high' : 
                     customer.churnProbability > 0.4 ? 'medium' : 'low';
    const riskText = customer.churnProbability > 0.7 ? 'High' : 
                    customer.churnProbability > 0.4 ? 'Medium' : 'Low';
    
    row.innerHTML = `
      <td>${customer.customerID}</td>
      <td>${customer.gender}</td>
      <td>${customer.tenure}</td>
      <td>${customer.Contract}</td>
      <td>$${customer.MonthlyCharges}</td>
      <td><span class="risk-value ${riskClass}">${riskText}</span></td>
    `;
    
    tbody.appendChild(row);
  });
}

// Utility functions
function formatPercentage(value) {
  return (value * 100).toFixed(1) + '%';
}

function formatCurrency(value) {
  return '$' + value.toLocaleString();
}

// Add smooth scrolling for better UX
function smoothScroll(target) {
  document.querySelector(target).scrollIntoView({
    behavior: 'smooth'
  });
}

// Add loading states for better UX
function showLoading(element) {
  element.innerHTML = '<div class="loading">Loading...</div>';
}

function hideLoading(element, content) {
  element.innerHTML = content;
}

// Error handling
function handleError(error, context) {
  console.error(`Error in ${context}:`, error);
  // Could add user-friendly error messages here
}

// Responsive chart handling
function handleResize() {
  // Force chart updates on window resize
  Chart.helpers.each(Chart.instances, function(instance) {
    instance.resize();
  });
}

window.addEventListener('resize', debounce(handleResize, 250));

// Debounce utility
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Add keyboard navigation
document.addEventListener('keydown', function(e) {
  if (e.ctrlKey || e.metaKey) {
    const keyMap = {
      '1': 'home',
      '2': 'demo',
      '3': 'insights',
      '4': 'analytics',
      '5': 'dataset',
      '6': 'deployment'
    };
    
    if (keyMap[e.key]) {
      e.preventDefault();
      document.querySelector(`[data-section="${keyMap[e.key]}"]`).click();
    }
  }
});

// Add animation on scroll (simple implementation)
function animateOnScroll() {
  const elements = document.querySelectorAll('.card, .metric-card, .team-member');
  
  elements.forEach(element => {
    const elementTop = element.getBoundingClientRect().top;
    const elementVisible = 150;
    
    if (elementTop < window.innerHeight - elementVisible) {
      element.style.opacity = '1';
      element.style.transform = 'translateY(0)';
    }
  });
}

// Initialize scroll animations
elements = document.querySelectorAll('.card, .metric-card, .team-member');
elements.forEach(element => {
  element.style.opacity = '0';
  element.style.transform = 'translateY(20px)';
  element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
});

window.addEventListener('scroll', debounce(animateOnScroll, 100));
window.addEventListener('load', animateOnScroll);

// Demo data presets
const demoPresets = {
  highRisk: {
    gender: 'Female',
    SeniorCitizen: '1',
    Partner: 'No',
    Dependents: 'No',
    tenure: '6',
    MonthlyCharges: '95.50',
    Contract: 'Month-to-month',
    InternetService: 'Fiber optic',
    PaymentMethod: 'Electronic check'
  },
  lowRisk: {
    gender: 'Male',
    SeniorCitizen: '0',
    Partner: 'Yes',
    Dependents: 'Yes',
    tenure: '60',
    MonthlyCharges: '45.20',
    Contract: 'Two year',
    InternetService: 'DSL',
    PaymentMethod: 'Credit card (automatic)'
  }
};

// Add preset buttons to demo
function addPresetButtons() {
  const form = document.getElementById('churnForm');
  if (!form) return;
  
  const presetContainer = document.createElement('div');
  presetContainer.className = 'preset-buttons';
  presetContainer.style.marginBottom = 'var(--space-16)';
  
  presetContainer.innerHTML = `
    <button type="button" class="btn btn--secondary btn--sm" onclick="loadPreset('highRisk')">Load High Risk Example</button>
    <button type="button" class="btn btn--secondary btn--sm" onclick="loadPreset('lowRisk')">Load Low Risk Example</button>
  `;
  
  form.insertBefore(presetContainer, form.firstChild);
}

function loadPreset(presetName) {
  const preset = demoPresets[presetName];
  if (!preset) return;
  
  const form = document.getElementById('churnForm');
  Object.keys(preset).forEach(key => {
    const input = form.querySelector(`[name="${key}"]`);
    if (input) {
      input.value = preset[key];
    }
  });
}

// Add preset buttons after DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(addPresetButtons, 100);
});

// Export functions for global access
window.loadPreset = loadPreset;