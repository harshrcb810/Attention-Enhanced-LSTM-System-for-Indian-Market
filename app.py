"""
StockSense Pro - Main Application Entry Point
Advanced AI-Powered Stock Analysis Platform

This is the main file that orchestrates all modular components.
All business logic has been extracted to separate modules:
- config.py: Configuration and constants
- services.py: Data fetching and business logic
- models.py: LSTM and ML models
- styles.py: CSS styling
- ui.py: Streamlit UI components
"""
import warnings
import ui
import config

warnings.filterwarnings('ignore')

def main():
    """Main application entry point"""
    # Initialize page
    ui.setup_page_config()
    ui.apply_custom_css()
    
    # Show header
    ui.show_header()
    
    # Show sidebar
    ui.show_sidebar()
    
    # Show trending stocks
    ui.show_trending_stocks()
    
    # Show stock selector
    ui.show_stock_selector()
    
    # Show recommendation if available
    ui.show_recommendation_result()
    
    # Show detailed analysis if stock has been analyzed
    if 'df' in ui.st.session_state and ui.st.session_state.df is not None:
        ui.st.markdown("---")
        
        # News Section
        ui.show_news_section()
        
        ui.st.markdown("---")
        
        # Technical Charts
        ui.show_technical_charts()
        
        ui.st.markdown("---")
        
        # Model Performance
        ui.show_model_performance()
        
        ui.st.markdown("---")
        
        # Backtest Section
        ui.show_backtest_section()
    
    # Sectoral Performance
    ui.show_sectoral_performance()
    
    # Footer
    ui.show_footer()

if __name__ == "__main__":
    main()
