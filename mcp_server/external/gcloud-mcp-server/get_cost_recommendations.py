#!/usr/bin/env python3
"""
Get All Cost Optimization Recommendations

This script queries ALL cost optimization recommenders across multiple locations
to give you a comprehensive view of cost-saving opportunities.
"""

import subprocess
import json
import sys

def run_gcloud(args):
    """Run a gcloud command and return parsed JSON output."""
    try:
        result = subprocess.run(
            ["gcloud"] + args,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return []
        return []
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return []

def get_all_cost_recommendations(project_id):
    """Get all cost optimization recommendations for a project."""
    
    print(f"🔍 Scanning project '{project_id}' for cost optimization opportunities...\n")
    
    # Define recommenders and their locations
    recommenders = [
        {
            "name": "Idle VM Instances",
            "id": "google.compute.instance.IdleResourceRecommender",
            "locations": ["global"]
        },
        {
            "name": "Idle IP Addresses",
            "id": "google.compute.address.IdleResourceRecommender",
            "locations": ["global", "us-central1", "us-east1", "us-west1", "europe-west1", "europe-west2", "asia-east1"]
        },
        {
            "name": "Idle Persistent Disks",
            "id": "google.compute.disk.IdleResourceRecommender",
            "locations": ["global"]
        },
        {
            "name": "Idle Cloud SQL Instances",
            "id": "google.cloudsql.instance.IdleRecommender",
            "locations": ["us-central1", "us-east1", "europe-west1", "europe-west2", "asia-east1"]
        },
    ]
    
    all_recommendations = []
    total_savings = 0.0
    
    for recommender in recommenders:
        print(f"📊 Checking: {recommender['name']}...")
        recommender_recs = []
        
        for location in recommender['locations']:
            recs = run_gcloud([
                "recommender", "recommendations", "list",
                f"--project={project_id}",
                f"--location={location}",
                f"--recommender={recommender['id']}",
                "--format=json"
            ])
            
            if recs:
                for rec in recs:
                    rec['_recommender_name'] = recommender['name']
                    rec['_location'] = location
                    recommender_recs.append(rec)
        
        if recommender_recs:
            print(f"   ✅ Found {len(recommender_recs)} recommendation(s)")
            all_recommendations.extend(recommender_recs)
            
            # Calculate savings
            for rec in recommender_recs:
                if 'primaryImpact' in rec and 'costProjection' in rec['primaryImpact']:
                    cost = rec['primaryImpact']['costProjection']['cost']
                    units = float(cost.get('units', 0))
                    nanos = float(cost.get('nanos', 0))
                    monthly_savings = abs(units + (nanos / 1e9))
                    total_savings += monthly_savings
        else:
            print(f"   ℹ️  No recommendations")
    
    # Display summary
    print("\n" + "=" * 70)
    print("COST OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"\n💰 Total Monthly Savings Potential: ${total_savings:.2f}")
    print(f"📋 Total Recommendations: {len(all_recommendations)}\n")
    
    if all_recommendations:
        print("Detailed Recommendations:")
        print("-" * 70)
        
        for i, rec in enumerate(all_recommendations, 1):
            # Extract key information
            content = rec.get('content', {})
            overview = content.get('overview', {})
            resource_name = overview.get('resourceName', 'Unknown')
            location = rec.get('_location', 'Unknown')
            recommender_name = rec.get('_recommender_name', 'Unknown')
            description = rec.get('description', 'No description')
            
            # Calculate savings
            monthly_savings = 0
            if 'primaryImpact' in rec and 'costProjection' in rec['primaryImpact']:
                cost = rec['primaryImpact']['costProjection']['cost']
                units = float(cost.get('units', 0))
                nanos = float(cost.get('nanos', 0))
                monthly_savings = abs(units + (nanos / 1e9))
            
            print(f"\n[{i}] {recommender_name}")
            print(f"    Resource: {resource_name}")
            print(f"    Location: {location}")
            print(f"    💵 Monthly Savings: ${monthly_savings:.2f}")
            print(f"    📝 {description}")
            
            # Show recommended action
            recommended_action = overview.get('recommendedAction', 'N/A')
            if recommended_action != 'N/A':
                print(f"    ⚡ Action: {recommended_action}")
        
        print("\n" + "=" * 70)
        print("💡 To get details on a specific recommendation, use:")
        print("   gcloud recommender recommendations describe RECOMMENDATION_ID \\")
        print("     --project=PROJECT --location=LOCATION --recommender=RECOMMENDER_ID")
    else:
        print("✅ Great! No cost optimization recommendations found.")
        print("   Your resources are being used efficiently.")
    
    print("\n")

if __name__ == "__main__":
    # Get project ID from gcloud config or argument
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
    else:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True
        )
        project_id = result.stdout.strip()
    
    if not project_id:
        print("❌ Error: Could not determine project ID")
        print("Usage: python get_cost_recommendations.py [PROJECT_ID]")
        sys.exit(1)
    
    get_all_cost_recommendations(project_id)
