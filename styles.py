"""
Styles Module
All CSS styling and frontend appearance
"""

def get_custom_css():
    """Return all custom CSS for the application"""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #0f1419 0%, #1a2332 25%, #2d3e50 50%, #34495e 75%, #2c3e50 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .main-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, rgba(0, 123, 255, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%);
            z-index: 1;
        }
        .main-header > * { position: relative; z-index: 2; }
        
        .trending-card {
            background: linear-gradient(135deg, rgba(15, 20, 25, 0.9) 0%, rgba(26, 35, 50, 0.9) 50%, rgba(45, 62, 80, 0.9) 100%);
            padding: 1.8rem;
            border-radius: 16px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .trending-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(0, 123, 255, 0.3);
            border-color: rgba(0, 123, 255, 0.5);
        }
        
        .search-container {
            background: linear-gradient(135deg, rgba(15, 20, 25, 0.95) 0%, rgba(26, 35, 50, 0.95) 50%, rgba(45, 62, 80, 0.95) 100%);
            padding: 3.5rem;
            border-radius: 24px;
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.4);
            text-align: center;
            margin: 2rem 0;
            backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: white;
            position: relative;
            overflow: hidden;
        }
        .search-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, rgba(0, 123, 255, 0.05) 0%, rgba(0, 200, 255, 0.05) 100%);
            z-index: 1;
        }
        .search-container > * { position: relative; z-index: 2; }
        
        .sidebar-section {
            background: linear-gradient(135deg, rgba(15, 20, 25, 0.8) 0%, rgba(26, 35, 50, 0.8) 100%);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0, 123, 255, 0.3);
            color: white !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 123, 255, 0.1);
            transition: all 0.3s ease;
        }
        .sidebar-section:hover {
            border-color: rgba(0, 123, 255, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 123, 255, 0.2);
        }
        
        .recommendation-result {
            padding: 3rem;
            border-radius: 24px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .buy-result { background: linear-gradient(135deg, #00C851 0%, #007E33 50%, #005722 100%); color: white; }
        .hold-result { background: linear-gradient(135deg, #ffbb33 0%, #e6a000 50%, #cc8800 100%); color: white; }
        .sell-result { background: linear-gradient(135deg, #ff4444 0%, #cc0000 50%, #990000 100%); color: white; }
        
        .chart-container {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 123, 255, 0.1);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 20, 25, 0.95) 0%, rgba(26, 35, 50, 0.95) 50%, rgba(34, 45, 60, 0.95) 100%);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(0, 123, 255, 0.2);
        }
        
        .sidebar-section h1, .sidebar-section h2, .sidebar-section h3, .sidebar-section h4,
        .sidebar-section p, .sidebar-section li, .sidebar-section blockquote,
        .sidebar-section small, .sidebar-section span, .sidebar-section div {
            color: white !important;
            font-weight: 400;
        }
        .sidebar-section h4 {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            color: #00c8ff !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4);
        }
        
        .main-header h1 {
            font-weight: 700;
            font-size: 2.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #00c8ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .market-index-card, .model-info-card {
            background: linear-gradient(135deg, rgba(15, 20, 25, 0.9) 0%, rgba(26, 35, 50, 0.9) 50%, rgba(45, 62, 80, 0.9) 100%);
            padding: 1.8rem;
            border-radius: 16px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
        }
        .market-index-card:hover, .model-info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(0, 123, 255, 0.3);
            border-color: rgba(0, 123, 255, 0.5);
        }
        
        .fade-in {
            opacity: 0;
            animation: fadeIn 0.8s ease-in forwards;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(0, 123, 255, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(0, 123, 255, 0.3);
            text-align: center;
            margin: 0.5rem 0;
        }
    </style>
    """
