from flask import Flask, request, jsonify
from supabase import create_client, Client
import os

# Initialize Flask app
application = Flask(__name__)

# Supabase credentials
SUPABASE_URL = "https://ismwspsxylmaatsdbvkt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlzbXdzcHN4eWxtYWF0c2Ridmt0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MDQxNzA4MCwiZXhwIjoyMDU1OTkzMDgwfQ.mt6aVhrUEha33VBXhkN6ABHt-jvaGzpkWG3tAxzcmKI"
SUPABASE_ANALYSES_TABLE = "analyses"
SUPABASE_UPLOADS_TABLE = "uploads"
SUPABASE_RELATIONSHIPS_TABLE = "relationships"


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@application.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    user_id = request.form.get('user_id')
    analysis_id = request.form.get('analysis_id')
    
    if not user_id or not analysis_id:
        return jsonify({"error": "Missing user_id or analysis_id"}), 400
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Update analysis record
        update_response = supabase.table(SUPABASE_ANALYSES_TABLE).update({
            "status": "analyzed",
            "summary": "",
            "red_flags": 0,
            "green_flags": 0,
            "personality_summaries": "[]",
            "flags": "{\"red\": [], \"green\": []}",
            "timeline": "[]"
        }).eq("id", analysis_id).execute()
        
        if "error" in update_response and update_response["error"]:
            return jsonify({"error": f"Failed to update analysis record: {update_response['error']['message']}"}), 500
        
        # Update upload status
        upload_update = supabase.table(SUPABASE_UPLOADS_TABLE).update({"status": "processed"}).eq("id", analysis_id).execute()
        if "error" in upload_update and upload_update["error"]:
            return jsonify({"error": f"Failed to update upload status: {upload_update['error']['message']}"}), 500
        
        # Update relationship status
        relationship_update = supabase.table(SUPABASE_RELATIONSHIPS_TABLE).update({
            "status": "analyzed",
            "summary": ""
        }).eq("upload_id", analysis_id).eq("analysis_id", analysis_id).execute()
        
        if "error" in relationship_update and relationship_update["error"]:
            return jsonify({"error": f"Failed to update relationship status: {relationship_update['error']['message']}"}), 500
        
        return jsonify({
            "success": True,
            "message": "Chat processed successfully",
            "analysis": {
                "incidents": 0,
                "red_flags": 0,
                "green_flags": 0,
                "partners": []
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    application.run(debug=True)
