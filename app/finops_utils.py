
"""
FinOps Utilities
----------------
Helper functions for normalizing and calculating cloud cost savings
from Google Cloud Recommender API data.
"""

def parse_savings_from_recommender(recommendation_json: dict) -> dict:
    """
    Extracts cost savings from a single recommendation JSON object.
    
    Args:
        recommendation_json (dict): The raw JSON of a recommendation.
        
    Returns:
        dict: {
            "currency": str,
            "monthly_savings": float
        }
    """
    primary_impact = recommendation_json.get("primaryImpact", {})
    if primary_impact.get("category") != "COST":
        return {"currency": "USD", "monthly_savings": 0.0}
        
    cost_projection = primary_impact.get("costProjection", {}).get("cost", {})
    currency_code = cost_projection.get("currencyCode", "USD")
    
    # "units" are whole units of currency (negative for savings)
    # "nanos" are 10^-9 units (negative for savings)
    units = int(cost_projection.get("units", 0))
    nanos = int(cost_projection.get("nanos", 0))
    
    # Formula: abs(units + nanos / 10^9)
    total_savings = abs(units + (nanos / 1_000_000_000))
    
    return {
        "currency": currency_code,
        "monthly_savings": round(total_savings, 2)
    }

def aggregate_savings(recommendations: list) -> dict:
    """
    Aggregates savings from a list of recommendations, grouped by currency.
    """
    totals = {}
    
    for rec in recommendations:
        savings_data = parse_savings_from_recommender(rec)
        currency = savings_data["currency"]
        amount = savings_data["monthly_savings"]
        
        if currency not in totals:
            totals[currency] = 0.0
        totals[currency] += amount
        
    # Format for display
    return {k: round(v, 2) for k, v in totals.items()}
