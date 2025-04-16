from flask import Flask, request, jsonify
import datetime
import threading
import tempfile
import os
import json
import random
import time
from tqdm import tqdm
import bisect
from dateutil import parser
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, TypedDict
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client, Client
import logging

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_ANALYSES_TABLE = "analyses"
SUPABASE_UPLOADS_TABLE = "uploads"
SUPABASE_RELATIONSHIPS_TABLE = "relationships"

main_model = "gpt-4.1"
format_model = "gpt-4.1-mini"

# Set up Flask app
application = Flask(__name__)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure maximum content length to 50MB
application.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary libraries for text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define models
class Tone(str, Enum):
    negative = 'negative'
    neutral = 'neutral'
    positive = 'positive'

class Intensity(str, Enum):
    low = 'low'
    high = 'high'

class Incident(BaseModel):
    start_time: str = Field(description="The exact timestamp when the conflict begins.")
    end_time: str = Field(description="The exact timestamp when the conflict concludes.")
    title: str = Field(description="A title that breifly descripe the conflict.")
    description: str = Field(description="Descripe breifly the conflict.")
    tone: Tone = Field(description="Can be ['Negative', 'Neutral', 'Positive']")
    intensity: Intensity = Field(description="Can be ['High' or 'Low']")

class ExtractedIncidents(BaseModel):
    Incident: List[Incident]

class Pillar(str, Enum):
    criticism = "criticism"
    defensiveness = "defensiveness"
    contempt = "contempt"
    stonewalling = "stonewalling"

class RelationshipIncident(BaseModel):
    id: int = Field(description="The unique identifier for the incident")
    title: str = Field(description="The title of the incident")
    rating: int = Field(description="The intensity rating of the incident on a scale from 1 to 10")
    objective_opinion: str = Field(description="An objective analysis of the incident, including fault and corrective actions")
    pillars: List[Pillar] = Field(description="The 'Four Horsemen' behaviors identified in the incident")

class Partner(BaseModel):
    name: str = Field(description="The name of the partner")
    personalityReport: str = Field(description="A detailed personality report of the partner")

class StructuredRelationshipSummary(BaseModel):
    overview: str = Field(description="Overview of the relationship dynamics and patterns")
    problems: str = Field(description="Most significant problems in the relationship, with examples")
    solutions: str = Field(description="Potential solutions for the identified problems")
    positives: str = Field(description="Positive feedback and relationship strengths") 
    conclusion: str = Field(description="Conclusion summarizing the relationship assessment")

class Relationship(BaseModel):
    Incidents: List[RelationshipIncident] = Field(description="A list of incidents that occurred in the relationship")
    RelationshipReport: str = Field(description="A detailed report of the relationship, including significant problems and solutions")
    PartnerReports: List[Partner] = Field(description="A list of personality reports for each partner")

# Define processing status enum
class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    error = "error"
    ready = "ready"

def update_analysis_status(analysis_id, status, upload_id=None, error_message=None):
    """
    Update the status of an analysis in the Supabase table.
    Also updates the relationship status when analysis status is set to processing.
    """
    update_data = {"status": status}
    logger.info(f"Error log entered update_analysis_status with analysis_id: {analysis_id}, status: {status}, error_message: {error_message}")
    print(f"Error print entered update_analysis_status with analysis_id: {analysis_id}, status: {status}, error_message: {error_message}")

    # If there's an error message, include it
    if error_message and status == ProcessingStatus.error:
        update_data["error_message"] = error_message
    
    try:
        update_response = supabase.table(SUPABASE_ANALYSES_TABLE).update(update_data).eq("id", analysis_id).execute()
        
        if "error" in update_response and update_response["error"]:
            logger.info(f"Failed to update analysis status: {update_response['error']['message']}")
            return False

        # If status is "processing" and upload_id is provided, also update relationship status
        if status == ProcessingStatus.processing and upload_id:
            relationship_update = supabase.table(SUPABASE_RELATIONSHIPS_TABLE).update({
                "status": "processing"
            }).eq("upload_id", upload_id).eq("analysis_id", analysis_id).execute()
            
            if "error" in relationship_update and relationship_update["error"]:
                logger.info(f"Failed to update relationship status: {relationship_update['error']['message']}")
                # Continue anyway, as this is not critical
        
        return True
    except Exception as e:
        logger.info(f"Error updating status: {str(e)}")
        return False

def parse_datetime(datetime_str):
    """
    Attempts to parse the given datetime string into a datetime object.
    Uses the dateutil library for broad format support.
    Returns None if the format could not be recognized.
    """
    try:
        return parser.parse(datetime_str)
    except (ValueError, TypeError):
        return None

def parse_chat_history(chat_history):
    """
    Parses the chat history into a sorted list of (datetime_object, message).
    Each line in chat_history is assumed to be in the format:
        [timestamp] the chat message...
    Returns a list sorted by datetime_object.
    """
    chat_lines = chat_history.strip().split("\n")
    parsed_messages = []

    for line in chat_lines:
        # Each line should contain the timestamp inside brackets [...]
        if line.startswith("[") and "]" in line:
            timestamp_part, message = line.split("]", 1)
            timestamp_str = timestamp_part.strip("[")
            dt = parse_datetime(timestamp_str)
            if dt is not None:
                parsed_messages.append((dt, message.strip()))
            # If dt is None, we skip and you could log a warning here if desired

    # Sort parsed messages by datetime
    parsed_messages.sort(key=lambda x: x[0])
    return parsed_messages

def get_chat_snippets_for_incident(incident, parsed_messages):
    """
    Given an incident (with 'start_time' and 'end_time')
    and a sorted list of (datetime_object, message),
    returns the snippet of messages that fall within the incident's time window.
    """
    start_dt = parse_datetime(incident.get("start_time"))
    end_dt = parse_datetime(incident.get("end_time"))

    if not start_dt or not end_dt:
        # If we can't parse the incident's time bounds, return empty snippet.
        return ""

    # Using bisect, find the start and end indices of the relevant messages
    start_index = bisect.bisect_left(parsed_messages, (start_dt, ""))
    end_index = bisect.bisect_right(parsed_messages, (end_dt, "\uffff"))

    # Now slice out the range of messages
    relevant_messages = parsed_messages[start_index:end_index]

    # Build the snippet
    snippet_lines = []
    for dt, msg in relevant_messages:
        snippet_lines.append(f"[{dt.strftime('%X')}] {msg}")

    return "\n".join(snippet_lines)

def split_into_batches(lst, batch_size):
    # Use list comprehension to create chunks of size batch_size
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def incidentToStr(incident):
    output = ""
    output += f"Incident ID: {incident['incident_id']}\n"
    output += f"Title: {incident['title']}\n"
    output += f"Description: {incident['description']}\n"
    output += f"Chat snippet:\n {incident['chat_snippet']}\n"
    return output

def listToPrompt(incidents):
    output = ""
    for inc in incidents:
        output += incidentToStr(inc) + "\n"
    return output

def generate_comprehensive_summary(string_list, llm_client):
    # Combine all strings with clear separators
    combined_text = "\n\n=== String Boundary ===\n\n".join(string_list)
    
    prompt = f"""
    Below are multiple text segments separated by "=== String Boundary ===".
    These segments may contain overlapping information and unique details.
    
    {combined_text}
    
    Create a single comprehensive report that:
    1. Captures ALL unique information from ALL segments
    2. Contains NO repetition of information
    3. Is well-structured and coherent
    4. Make it organized and easy to read.
    
    Focus on identifying new information in each segment while 
    avoiding redundancy in the final report.
    """
    
    response = llm_client.invoke(prompt)
    return response

def aggregate_relationships(relationships, idToIncident):
    """
    Given a list of Relationship objects, aggregate:
      1. A list of all incidents.
      2. A list of all relationship summaries.
      3. A list of partner summaries, grouped by partner name.
    """
    
    # List to store all incidents
    all_incidents = []
    # List to store relationship summaries
    relationship_summaries = []
    # Dictionary to store partner summaries keyed by partner name
    partner_summaries = {}

    # Iterate through each Relationship
    for relationship in relationships:
        # Collect the Relationship summary
        relationship_summaries.append(relationship.RelationshipReport)
        
        # Collect Partner summaries
        for partner in relationship.PartnerReports:
            if partner.name not in partner_summaries:
                partner_summaries[partner.name] = []
            partner_summaries[partner.name].append(partner.personalityReport)
        
        # Collect Incidents
        for incident in relationship.Incidents:
            # Get the incident data from idToIncident
            if str(incident.id) in idToIncident:
                inc_data = idToIncident[str(incident.id)]
                
                # Convert the Incident object to a dict or minimal structure
                incident_data = {
                    'id': incident.id,
                    'title': incident.title,
                    'rating': incident.rating,
                    'description': inc_data.get("description", ""),
                    'objective_opinion': incident.objective_opinion,
                    'start_time': inc_data.get("start_time", ""),
                    'end_time': inc_data.get("end_time", ""),
                    # Convert pillars (which are Enums) to string names
                    'pillars': [pillar.name for pillar in incident.pillars],
                    'chat_snippet': inc_data.get("chat_snippet", "")
                }
                all_incidents.append(incident_data)

    # Build a final aggregated dictionary
    aggregated_data = {
        'incidents': all_incidents,
        'relationship_summaries': relationship_summaries,
        'partner_summaries': partner_summaries
    }
    
    return aggregated_data

def process_chat_async(tmp_file_path, api_key):
    """
    Process chat file asynchronously and save results to a JSON file.
    """
    user_id = None
    analysis_id = None
    upload_id = None
    try:
        # Read the content
        with open(tmp_file_path, 'r', encoding='utf-8') as f:
            chat_history = f.read()

        #load the user_id and analysis_id from the file
        with open(tmp_file_path+"maher", 'r') as f:
            user_id = f.readline().strip().split(":")[1].strip()
            analysis_id = f.readline().strip().split(":")[1].strip()
            upload_id = f.readline().strip().split(":")[1].strip()

        # Update status to processing
        update_analysis_status(analysis_id, ProcessingStatus.processing, upload_id)
        
        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=14000, chunk_overlap=400)
        docs = [Document(page_content=chat_history)]
        all_splits = text_splitter.split_documents(docs)
        
        # Initialize LLMs
        llm = ChatOpenAI(temperature=0, model=main_model, api_key=api_key).with_structured_output(
            ExtractedIncidents
        )
        llm_unstructured = ChatOpenAI(temperature=0, model=main_model, api_key=api_key)
        llm_formater = ChatOpenAI(temperature=0, model=format_model, api_key=api_key).with_structured_output(
            Relationship
        )
        
        # Define the tagging prompt
        tagging_prompt = ChatPromptTemplate.from_template(
            """
            You are an AI tasked with analyzing chat histories to identify significant conflicts between partners. A significant incident involves a clear disagreement, argument, or period of notable tension that potentially impacts the relationship negatively. Your goal is to extract ONLY these significant incidents, ignoring minor or quickly resolved issues. The time in the chat is written in this format [DD/MM/YYYY, H:mm:ss AM/PM].

            For each detected significant incident, provide the following:
                1. Title: A title that briefly describes the incident.
                2. Description: Briefly describe the incident, focusing on the core conflict.
                3. Start Time: The exact timestamp when the incident begins, using the format "%d/%m/%Y, %I:%M:%S %p".
                4. End Time: The exact timestamp when the incident concludes, using the format "%d/%m/%Y, %I:%M:%S %p".
                5. Tone: Overall tone of the incident ["Negative", "Neutral", "Positive"]. Negative is expected for conflicts.
                6. Intensity: Overall intensity of the conflict ["High" or "Low"]. Focus extraction on HIGH intensity incidents.

            Guidelines for Identifying SIGNIFICANT Incidents:
                •   Look for interactions with clear, sustained tension, escalation, strong negative emotions (anger, frustration, contempt, accusations), or discussion of core relationship problems.
                •   An incident should typically span multiple message exchanges and show difficulty in resolution.
                •   Prioritize conflicts that reveal underlying relationship dynamics or recurring issues.

            What NOT to Extract (Examples of NON-Significant Incidents):
                •   Casual conversation, jokes, planning, or lighthearted banter.
                •   Sarcasm, inside jokes, or subtle humor.
                •   Simple statements of fact or preference without emotional charge (e.g., "I don't like that movie").
                •   Minor disagreements or misunderstandings that are clarified and resolved within 1-2 messages.
                •   Brief moments of annoyance or sarcasm that don't escalate or lead to a larger argument.
                •   Playful teasing, even if slightly edgy.

            Tone Definitions (Use to assess incident, but TONE alone doesn't define significance):
                •   Negative: Conveys dissatisfaction, conflict, emotional distress (e.g., angry, frustrated, defensive, dismissive, contemptuous).
                •   Neutral: Impartial, factual, calm (e.g., informative, objective, polite).
                •   Positive: Optimistic, supportive, connecting (e.g., friendly, empathetic, cheerful).

            If no significant incident meeting these criteria is detected in the passage, return an empty list:
            []

            By focusing only on significant conflicts, ensure the output is concise and relevant to meaningful relational analysis.
            Only extract the properties mentioned in the 'ExtractedIncidents' function.

            Passage:
            {input}
            """
        )
        
        # Process in batches
        batch_size = 200  # Adjust batch size to fit within rate limits
        
        # Process the content in chunks
        results = []
        for chunk in all_splits:
            prompt = tagging_prompt.invoke({"input": chunk.page_content})
            response = llm.invoke(prompt)
            results.append(response)
            
        # Filter out empty results
        filtered_results = [res for res in results if len(res.Incident) > 0]
        
        # Process incidents
        incidents = []
        for f in filtered_results:
            for i in f.Incident:
                if i.intensity == "high":
                    incidents.append(i.model_dump())
        
        # Update progress
        update_analysis_status(analysis_id, ProcessingStatus.processing, upload_id)
        
        # Parse chat history
        parsed_messages = parse_chat_history(chat_history)
        
        # Process incidents to add chat snippets
        idToIncident = {}
        for inc in incidents:
            chat_snippet = get_chat_snippets_for_incident(inc, parsed_messages)
            inc["chat_snippet"] = chat_snippet
            inc["incident_id"] = random.randint(10000, 99999)
            idToIncident[str(inc["incident_id"])] = inc
        
        # Split incidents into batches
        batches = split_into_batches(incidents, 40)
        
        # Define relationship analysis prompt
        relationship_prompt = ChatPromptTemplate.from_template(
            """
            You are a relationship therapist in the style of Esther Perel, John Gottman, or Harville Hendrix. Your main objective is to objectively analyze chat snippets between two partners, identify common patterns and problems, and provide direct, concise feedback without diplomatic cushioning. Do not attempt to sound "nice." Then, offer potential solutions and areas for improvement. the goal is to fix the problems so don't be pessimistic.

            Given the following chat snippets, please extract the following:
                1.	For Each Incident:
                •	ID: The unique identifier for the incident
                •	Title: A title that briefly describes the incident.
                •	Intensity Rating: How significant the incident is to the relationship compared to other incidents, Intensity value is [1 to 10], 10 being most significant.
                •	Objective Opinion: as an objective person report your analysis of the incident, including who is at fault if necessary, and why is faulty and what is the right things that should have been done.
                •	Pillars: According to John Gottman, specify if the incident shows any of the "Four Horsemen": criticism, defensiveness, contempt, or stonewalling.
                2.	For All Incidents:
                •	Report of the Relationship: List the (Most Significant Problems) in details, for each significant problem you must support it by real examples that occurred in the chat to support your analysis, and include the Incident ID as reference to each example. Then provide (potential solutions), and any positive feedback that may exist (Positive Feedback & Strengths), support it with examples as well. Finally, provide a detailed overview and analysis of the relationship and don't be vague (verview of the relationship) then finally provide (conclusion).
                •	For each partner, provide his/her name and a personality report. and use the name in the chat snippets.    
                    •	Personality report: You are a psychologist specializing in personality assessment. Provide a concise yet thorough analysis by detailing each individual's communication style and emotional patterns (how they express themselves, handle conflict, and show emotions), exploring any insecurities and motivations (what appears to drive or concern them), and highlighting both their positive and negative qualities (strengths, weaknesses, and potential blind spots). Conclude with a brief overview of how these traits may shape their interactions and offer practical suggestions for improving their dynamic.
            chat snippets:
            {input}
            """
        )

        logger.info(f"batches: {batches[:1]}")
        
        # Format and process batches
        therapy_responses = []
        for batch in batches:
            prompt = relationship_prompt.invoke({"input": listToPrompt(batch)})
            response = llm_unstructured.invoke(prompt)
            therapy_responses.append(response.content)
        
        # Update progress
        update_analysis_status(analysis_id, ProcessingStatus.processing, upload_id)
        
        # Format therapy responses
        format_prompt = ChatPromptTemplate.from_template(
            """
            Rewrite the following into the format, use the exact names in the chat snippets, and don't miss any details or change anything.
            {input}
            """
        )
        
        formatted_responses = []
        for response in therapy_responses:
            prompt = format_prompt.invoke({"input": response})
            formatted = llm_formater.invoke(prompt)
            formatted_responses.append(formatted)
        
        logger.info(f"Formatted responses: {formatted_responses}")

        # Aggregate the therapy results
        aggregated_result = aggregate_relationships(formatted_responses, idToIncident)
        
        # Generate comprehensive summaries
        relationship_summary = generate_comprehensive_summary(
            aggregated_result["relationship_summaries"], 
            llm_unstructured
        ).content
        
        # Create a structured summary formatter
        structure_summary_prompt = ChatPromptTemplate.from_template(
            """
            Rewrite the following into the format, don't summarize or change anything, use the exact markdown format, just copy the exact text in the provided format.
            {input}
            """
        )
        
        # Initialize llm with structured output for summary
        llm_summary_formatter = ChatOpenAI(temperature=0, model=main_model, api_key=api_key).with_structured_output(
            StructuredRelationshipSummary
        )
        
        # Format the relationship summary into structured sections
        structured_summary_prompt = structure_summary_prompt.invoke({"input": relationship_summary})
        structured_summary = llm_summary_formatter.invoke(structured_summary_prompt)
        
        partner_keys = list(aggregated_result["partner_summaries"].keys())
        
        # Prepare the final response
        final_result = {
            "incidents": aggregated_result["incidents"],
            "relationship_summary": relationship_summary,
            "structured_summary": {
                "overview": structured_summary.overview,
                "problems": structured_summary.problems,
                "solutions": structured_summary.solutions,
                "positives": structured_summary.positives,
                "conclusion": structured_summary.conclusion
            }
        }

        if len(partner_keys) >= 2:
            partner0_personality = generate_comprehensive_summary(
                aggregated_result["partner_summaries"][partner_keys[0]], 
                llm_unstructured
            ).content
            

            partner1_personality = generate_comprehensive_summary(
                aggregated_result["partner_summaries"][partner_keys[1]], 
                llm_unstructured
            ).content

            final_result[partner_keys[0]] = partner0_personality
            final_result[partner_keys[1]] = partner1_personality
        else:
            partner0_personality = ""
            partner1_personality = ""
            
            if len(partner_keys) == 1:
                partner0_personality = generate_comprehensive_summary(
                    aggregated_result["partner_summaries"][partner_keys[0]], 
                    llm_unstructured
                ).content
                final_result[partner_keys[0]] = partner0_personality

        # Count red and green flags (based on pillars)
        red_flags = len([incident for incident in aggregated_result["incidents"] if incident["pillars"]])
        green_flags = len([incident for incident in aggregated_result["incidents"] if not incident["pillars"]])
        
        final_result["red_flags"] = red_flags
        final_result["green_flags"] = green_flags
        
        # logging 
        logger.info(f"Final result: {final_result["relationship_summary"]}")
        logger.info(f"Structured summary overview: {structured_summary.overview[:100]}...")
        logger.info(f"Structured summary problems: {structured_summary.problems[:100]}...")
        logger.info(f"Structured summary solutions: {structured_summary.solutions[:100]}...")
        logger.info(f"Structured summary positives: {structured_summary.positives[:100]}...")
        logger.info(f"Structured summary conclusion: {structured_summary.conclusion[:100]}...")
        logger.info(f"partner0 summary: {partner0_personality}")
        logger.info(f"partner1 summary: {partner1_personality}")
        logger.info(f"Red flags: {red_flags}")
        logger.info(f"Green flags: {green_flags}")
        logger.info(f"Incidents: {final_result["incidents"]}")


        try:
            # Update analysis record - set status to ready
            update_response = supabase.table(SUPABASE_ANALYSES_TABLE).update({
                "status": ProcessingStatus.ready,
                "summary": final_result["relationship_summary"],
                "summary_structured": json.dumps({
                    "overview": structured_summary.overview,
                    "problems": structured_summary.problems,
                    "solutions": structured_summary.solutions,
                    "positives": structured_summary.positives,
                    "conclusion": structured_summary.conclusion
                }),
                "red_flags": red_flags,
                "green_flags": green_flags,
                "personality_summaries": json.dumps({
                    partner_keys[0]: partner0_personality,
                    partner_keys[1]: partner1_personality
                }) if len(partner_keys) >= 2 else json.dumps({}),
                "flags": json.dumps({
                    "red": [
                        {
                            "id": incident["id"],
                            "title": incident["title"],
                            "pillars": incident["pillars"]
                        }
                        for incident in aggregated_result["incidents"]
                        if incident["pillars"]
                    ],
                    "green": [
                        {
                            "id": incident["id"],
                            "title": incident["title"]
                        }
                        for incident in aggregated_result["incidents"]
                        if not incident["pillars"]
                    ]
                }),
                "timeline": json.dumps(final_result["incidents"])
            }).eq("id", analysis_id).execute()
                
            if "error" in update_response and update_response["error"]:
                error_message = f"Failed to update analysis record: {update_response['error']['message']}"
                update_analysis_status(analysis_id, ProcessingStatus.error, upload_id, error_message)
                return
            
            # Update upload status
            upload_update = supabase.table(SUPABASE_UPLOADS_TABLE).update({"status": "processed"}).eq("id", upload_id).execute()
            if "error" in upload_update and upload_update["error"]:
                error_message = f"Failed to update upload status: {upload_update['error']['message']}"
                update_analysis_status(analysis_id, ProcessingStatus.error, upload_id, error_message)
                return
            
            # Update relationship status
            relationship_update = supabase.table(SUPABASE_RELATIONSHIPS_TABLE).update({
                "status": "analyzed",
                "summary": ""
            }).eq("upload_id", upload_id).eq("analysis_id", analysis_id).execute()
            
            if "error" in relationship_update and relationship_update["error"]:
                error_message = f"Failed to update relationship status: {relationship_update['error']['message']}"
                update_analysis_status(analysis_id, ProcessingStatus.error, upload_id, error_message)
                return
            
        except Exception as e:
            error_message = f"An unexpected error occurred during database updates: {str(e)}"
            update_analysis_status(analysis_id, ProcessingStatus.error, upload_id, error_message)

    except Exception as e:
        error_message = f"Error processing file: {str(e)}"
        print(error_message)
        if analysis_id:
            update_analysis_status(analysis_id, ProcessingStatus.error, upload_id, error_message)
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

    
@application.route('/analyze', methods=['POST'])
def analyze_chat():
    """
    Endpoint to receive and process a chat file asynchronously.
    Returns a 200 status code immediately after receiving the file.
    """
    logger.info(f" log entered update_analysis_status with analysis_id")
    print(f" print entered update_analysis_status with analysis_id")

    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    user_id = request.form.get('user_id')
    analysis_id = request.form.get('analysis_id')
    upload_id = request.form.get('upload_id')
    
    # Check if the file has content
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    
    logger.info(f"File received with user_id: {user_id}, analysis_id: {analysis_id}, upload_id: {upload_id}")

    # Get API key from request headers or environment
    api_key = request.headers.get('X-API-KEY', os.environ.get('OPENAI_API_KEY'))
    if not api_key:
        return jsonify({'error': 'API key is required'}), 401
    
    try:
        # Set initial status to pending
        update_analysis_status(analysis_id, ProcessingStatus.pending, upload_id)
        
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp:
            file.save(tmp.name)
            tmp_file_path = tmp.name
        
        logger.info(f"File saved to: {tmp_file_path}")

        # Add the user_id and analysis_id to the file
        with open(tmp_file_path+"maher", 'w') as f:
            f.write(f"user_id: {user_id}\n")
            f.write(f"analysis_id: {analysis_id}\n")
            f.write(f"upload_id: {upload_id}\n")
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=process_chat_async,
            args=(tmp_file_path, api_key)
        )
        thread.daemon = True
        thread.start()
            
        # Return immediate success response
        return jsonify({
            "status": "success", 
            "message": "File received and processing started in the background",
            "analysis_id": analysis_id,
            "upload_id": upload_id
        }), 200
        
    except Exception as e:
        error_message = str(e)
        # Set status to error
        update_analysis_status(analysis_id, ProcessingStatus.error, upload_id, error_message)
        return jsonify({'error': error_message}), 500
    finally:
        # Clean up temporary file if something went wrong before thread started
        if 'tmp_file_path' not in locals() and 'tmp' in locals():
            try:
                os.unlink(tmp.name)
            except:
                pass
            
@application.route('/', methods=['GET'])
def health_check():
    agent = request.headers.get('User-Agent', '')
    if 'ELB-HealthChecker' in agent:
        logger.info("Health check from ELB received")
    else:
        logger.info(f"Root path accessed by: {agent}")
    return jsonify({
    'status': 'healthy',
    'service': 'file-processing-api'
    }), 200
if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=8000, load_dotenv=True)
