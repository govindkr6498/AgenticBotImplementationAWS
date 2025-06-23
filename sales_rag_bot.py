import logging
from datetime import datetime
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional, List, Union
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import json
import pytz
import re  
import sqlite3
import boto3
from tabulate import tabulate
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from langchain.schema import Document
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field, EmailStr
from enum import Enum
from datetime import datetime, timedelta
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

class LeadInfo(BaseModel):
    """Model for lead information."""
    Name: str = Field(description="Full name of the lead")
    Company: str = Field(description="Company name")
    Email: EmailStr = Field(description="Email address of the lead")
    Phone: str = Field(description="Phone number of the lead")

class LeadCaptureState(Enum):
    """Enum for tracking lead capture state."""
    NO_INTEREST = "no_interest"
    INTEREST_DETECTED = "interest_detected"
    COLLECTING_INFO = "collecting_info"
    INFO_COMPLETE = "info_complete"
    AWAITING_MEETING_CONFIRMATION = "awaiting_meeting_confirmation"
    WAITING_MEETING_SLOT_SELECTION = "waiting_meeting_slot_selection"

class SalesforceAPI:
    """Salesforce API client for lead management."""
    def __init__(self):
        """Initialize Salesforce API client."""
        self.auth_url = "https://iqb4-dev-ed.develop.my.salesforce.com/services/oauth2/token"
        self.client_id = os.getenv("SF_CLIENT_ID")
        self.client_secret = os.getenv("SF_CLIENT_SECRET")
        self.access_token = None
        self.instance_url = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Salesforce and get access token."""
        try:
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }

            response = requests.post(self.auth_url, data=auth_data)
            response.raise_for_status()

            data = response.json()
            self.access_token = data.get("access_token")
            self.instance_url = data.get("instance_url")
            logger.info("Successfully authenticated with Salesforce")
            
        except Exception as e:
            logger.error(f"Salesforce authentication failed: {str(e)}")
            raise

    def create_lead(self, lead_info: Dict[str, str]) -> bool:
        """
        Create a new lead in Salesforce.
        
        Args:
            lead_info (Dict[str, str]): Lead information to create
            
        Returns:
            bool: True if lead was created successfully, False otherwise
        """
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()

            if any(value == "N/A" for value in lead_info.values()):
                logger.error("Cannot create lead with N/A values")
                return False

            lead_url = f"{self.instance_url}/services/data/v60.0/sobjects/Lead/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            sf_lead_payload = {
                "LastName": lead_info["Name"], 
                "Company": lead_info["Company"],
                "Email": lead_info["Email"],
                "Phone": lead_info["Phone"]
            }
            response = requests.post(
                lead_url,
                headers=headers,
                json=sf_lead_payload
            )
            if response.status_code == 201:
                lead_id = response.json().get("id")
                logger.info(f"lead_id: {lead_id}")
                logger.info(f"Lead created successfully: {response.json()}")
                return True, lead_id
            elif response.status_code == 400 and "DUPLICATES_DETECTED" in response.text:
                error_data = response.json()
                match_records = (
                    error_data[0]
                    .get("duplicateResult", {})
                    .get("matchResults", [])[0]
                    .get("matchRecords", [])
                )
                if match_records:
                    lead_id = match_records[0]["record"]["Id"]
                    logger.info(f"duplicate_lead_id: {lead_id}")
                    logger.info(f"Lead created successfully: {response.json()}")
                    return True, lead_id
            else:
                logger.info('fail Created')
            return "Would you like to meet a sales advisor this week?"
            
        except Exception as e:
            logger.error(f"Failed to create lead: {str(e)}")
            return False
    
    def create_meeting(self, lead_id: str, start_time_str: str) -> bool:
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()

            event_url = f"{self.instance_url}/services/data/v60.0/sobjects/Event/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            start_dt = datetime.strptime(start_time_str, "%H:%M")
            ist = pytz.timezone('Asia/Kolkata')
            start_dt_local = datetime.strptime(start_time_str, "%H:%M")
            today_local = datetime.now(ist).date()
            start_local_dt = ist.localize(datetime.combine(today_local, start_dt_local.time()))
            start_utc_dt = start_local_dt.astimezone(pytz.utc) + timedelta(hours=5) + timedelta(minutes=30)
            end_utc_dt = start_utc_dt + timedelta(minutes=30)

            start_datetime_iso = start_utc_dt.isoformat()
            end_datetime_iso = end_utc_dt.isoformat()

            event_payload = {
                "Subject": "Call with Sales Advisor",
                "StartDateTime": start_datetime_iso,
                "EndDateTime": end_datetime_iso,
                "OwnerId": "0055j00000BYNIBAA5",
                "WhoId": lead_id,
                "Location": "Virtual Call",
                "Description": "Scheduled via Agentic Bot"
            }

            response = requests.post(event_url, headers=headers, json=event_payload)
            if response.status_code == 201:
                meeting_id = response.json().get("id")
                logger.info(f"Meeting created successfully: {meeting_id}")
                return True
            else:
                logger.error(f"Failed to create meeting: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Exception while creating meeting: {str(e)}")
            return False

    def show_availableMeeting(self) -> Optional[List[str]]:
        start_times = set()
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()
            event_url = f"{self.instance_url}/services/data/v60.0/query?q=SELECT+StartDateTime,+EndDateTime+FROM+Event+WHERE+StartDateTime+=+TODAY"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            response = requests.get(event_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])

                fmt = "%H:%M"
                start_time = datetime.strptime("08:00", fmt)
                end_time = datetime.strptime("17:00", fmt)
                all_slots = set()
                current = start_time
                while current < end_time:
                    all_slots.add(current.strftime(fmt))
                    current += timedelta(minutes=30)               
                for event in records:
                    start = event.get("StartDateTime")
                    if start:
                        try:
                            dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f%z")
                            time_only = dt.strftime("%H:%M")
                            start_times.add(time_only)
                        except Exception as parse_err:
                            logger.warning(f"Failed to parse StartDateTime: {start} ({parse_err})")
                available_slots = sorted(all_slots - start_times)
                logger.info(f"Schedule  meeting slots: {start_times}")
                logger.info(f"Available meeting slots: {available_slots}")
                return available_slots
            else:
                logger.error(f"Failed to showing meeting Time: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Exception while showing meeting: {str(e)}")
            return False


class SalesRAGBot:
    def __init__(self, pdf_path: str, model_name: str = "gpt-3.5-turbo-0125"):
        """Initialize the Sales RAG Bot."""
        self.pdf_path = pdf_path
        self.model_name = model_name
        self._setup_environment()
        self._initialize_components()
        self.lead_state = LeadCaptureState.NO_INTEREST
        self.partial_lead_info = {}
        self.conversation_history = []
        self.salesforce = SalesforceAPI()
        self.awaiting_meeting_confirmation = False
        self.awaiting_meeting_slot_selection = False
        self.awaiting_meeting_response = False
        self.current_lead_id = None
        self.available_slots = []
        logger.info("SalesRAGBot initialized")

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.docs = []
        self.load_documents()

        logger.info("SalesRAGBot initialized")


    def _setup_environment(self) -> None:
        """Set up environment variables and API keys."""      
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        logger.info("Environment setup completed")


    def _initialize_components(self) -> None:
        """Initialize all necessary components for the chatbot."""
        try:
            self.llm = ChatOpenAI(model=self.model_name)
            self._load_pdf()
            self._setup_vector_store()          
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise


    def _load_pdf(self) -> None:
        """Load and split the PDF document."""
        try:
            loader = PyPDFLoader(self.pdf_path)
            raw_docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len,
            )
            
            self.docs = text_splitter.split_documents(raw_docs)
            logger.info(f"PDF loaded and split into {len(self.docs)} chunks")
                
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise


    def _setup_vector_store(self) -> None:
        """Set up the FAISS vector store."""
        try:
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(
                documents=self.docs,
                embedding=embeddings
            )
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise


    def load_documents(self):
        print("Starting load_documents")
        local_path = 'C:/Users/admin/Documents/Document/Bot/src/storeJsonRecord'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.docs = [Document(**d) for d in data]
                print(f"Loaded {len(self.docs)} docs from local JSON")
                logger.info(f"Loaded {len(self.docs)} documents from local storage.")
            except Exception as e:
                logger.error(f"Error loading local JSON: {e}")
                print("Error loading local JSON:", e)
                self.docs = []
        else:
            print("Local file not found, calling fetch_and_save_aws_data")
            self.fetch_and_save_aws_data(local_path=local_path)
            print("After fetch_and_save_aws_data")
            if os.path.exists(local_path):
                try:
                    with open("C:/Users/admin/Documents/Document/Bot/src/storeJsonRecord", 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.docs = [Document(**d) for d in data]
                    print(f"Loaded {len(self.docs)} docs after fetch")
                    logger.info(f"Loaded {len(self.docs)} documents from fetched data.")
                except Exception as e:
                    logger.error(f"Error loading JSON after fetching: {e}")
                    self.docs = []
                    print("Error after fetch:", e)
            print(f"self.docs : {self.docs}")


    def fetch_and_save_aws_data(self, local_path='C:/Users/admin/Documents/Document/Bot/src/storeJsonRecord'):
        bucket_name = 'accountrecord'
        key = 'emaarGroupRecords/b8db7d18-6340-4f1b-9bbb-a2736503cc98/1593091727-2025-06-13T07:17:56'
        try:
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            json_content = obj['Body'].read().decode('utf-8')
            logger.info("Fetched AWS file content from S3")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            logger.info(f"Saved raw JSON data locally at {local_path}")
            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                lines = json_content.strip().split('\n')
                data = [json.loads(line) for line in lines if line]

            if isinstance(data, dict):
                data = [self._flatten_dict(data)]
            elif isinstance(data, list):
                if data and isinstance(data[0], dict):
                    if any(isinstance(v, dict) for v in data[0].values()):
                        data = [self._flatten_dict(item) for item in data]
            else:
                data = []

            docs = []
            for record in data:
                area_code = int(record.get('Area_Code__c', 0))
                plot_number = int(record.get('Plot_Number__c', 0))
                unit_price = float(record.get('Unit_Price__c', 0))
                unit_no = int(record.get('Unit_No__c', 0))
                text = (
                    f"Name: {record.get('Name', 'N/A')}\n"
                    f"Area: {record.get('Area__c', 'N/A')}\n"
                    f"Area Code: {area_code} BHK\n"
                    f"Plot Number: {plot_number}\n"
                    f"Project Name: {record.get('Project_Name__c', 'N/A')}\n"
                    f"Project Type: {record.get('Project_Type__c', 'N/A')}\n"
                    f"Building Number: {int(record.get('Building_Number__c', 0))}\n"
                    f"Tower Name: {record.get('Tower_Name__c', 'N/A')}\n"
                    f"Unit No: {unit_no}\n"
                    f"Unit Type: {record.get('Unit_Type__c', 'N/A')}\n"
                    f"Unit Price: {unit_price}\n"
                    f"District: {record.get('District__c', 'N/A')}\n"
                )
                docs.append(Document(page_content=text, metadata={"source": "AWS_S3"}))
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump([d.dict() for d in docs], f)
            print(f"Saved {len(docs)} documents to local storage.")
            logger.info(f"Saved {len(docs)} documents locally at {local_path}")
            return docs 
        except Exception as e:
            logger.error(f"Error fetching/saving AWS data: {e}")
            return []


    def get_combined_context(self, query: str) -> str:
        self.fetch_and_save_aws_data() 
        local_path = 'C:/Users/admin/Documents/Document/Bot/src/storeJsonRecord'
        try:
            if os.path.exists(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                aws_context_str = json.dumps(data, indent=2)
            else:
                aws_context_str = "No AWS data available."
        except Exception as e:
            aws_context_str = "Error loading AWS data."
            logger.error(f"Error loading AWS data for context: {e}")
        doc_context = self._get_relevant_context(query)
        combined = f"Document Info:\n{doc_context}\n\nAWS Data:\n{aws_context_str}"
        return combined


    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context from the vector store."""
        try:
            docs = self.vector_store.similarity_search(query, k=5)
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""


    def _extract_lead_info(self, message: str) -> Optional[Dict[str, str]]:
        """Extract lead information from the message."""
        try:
            prompt = f"""Extract contact information from the following message. Return ONLY a JSON object with these exact fields if found:
            - Name (required; store full name here)
            - Company (required)
            - Email (required)
            - Phone (required)

            Rules:
            1. Extract information even if it's provided in different formats or variations
            2. For Name:
            - Look for patterns like "name is X", "I am X", "my name is X", "this is X", "I'm X"
            - If full name is provided, keep the full name in the Name field
            - Do not shorten or split the name
            - Also look for "I'm X" or just "X" if it appears to be a name
            3. For company:
            - Look for patterns like "I work at X", "my company is X", "I'm from X", "at X", "with X"
            - Also look for "company name is X", "organization is X"
            - Don't extract company names from general product mentions
            4. For email:
            - Look for patterns like "email is X", "my email is X", "contact me at X"
            - Also look for "reach me at X", "send to X"
            - Extract any valid email address format
            5. For phone:
            - Look for patterns like "number is X", "phone is X", "call me at X"
            - Also look for "reach me at X", "contact me at X"
            - Accept various phone formats (with/without country code, spaces, dashes)
            6. If multiple values are found for a field, use the most recent or most specific one
            7. If a field is not found, do not include it in the JSON
            8. Return null if no contact information is found
            9. Handle both single-line and multi-line information
            10. Look for information even if it's just the value without context (e.g., just an email address)  
            10. **Only provide the answer data available context from the PDF**
            11. if there is no relevant sytem context as per the question then tell me i dont have enough knowldege about it
            12. always check the relevance of system context and user query
            System Context: 
        
            Human: {message}       
            Message: {message}

            Return ONLY the JSON object or null, nothing else.
            
            """
                       
            response = self.llm.invoke(prompt)
            try:
                lead_data = json.loads(response.content)
                if lead_data:
                    normalized_data = {}
                    for field in ['Name', 'Email', 'Phone']:
                        if field in lead_data and lead_data[field] and lead_data[field] != "N/A":
                            normalized_data[field] = lead_data[field].strip()
                    
                    if normalized_data:
                        normalized_data['Company'] = 'Iquestbee Technology'
                        return normalized_data
            except json.JSONDecodeError:
                pass
            return None

        except Exception as e:
            logger.error(f"Error extracting lead info: {str(e)}")
            return None


    def _update_lead_state(self, message: str) -> None:
        """Update the lead capture state based on the message."""
        interest_indicators = [
            "schedule", "meeting", "interested", "pricing", "cost","interest",
            "sign up", "enroll", "register", "buy", "purchase","want","Desire"
        ]
        
        if self.lead_state == LeadCaptureState.NO_INTEREST:
            if any(indicator in message.lower() for indicator in interest_indicators):
                self.lead_state = LeadCaptureState.INTEREST_DETECTED
                logger.info("Interest detected in conversation")
        
        if self.lead_state in [LeadCaptureState.INTEREST_DETECTED, LeadCaptureState.COLLECTING_INFO]:
            lead_info = self._extract_lead_info(message)
            if lead_info:
                self.partial_lead_info.update(lead_info)
                self.lead_state = LeadCaptureState.COLLECTING_INFO
                
                if all(key in self.partial_lead_info and self.partial_lead_info[key] not in [None, "N/A"]
                    for key in ['Name', 'Email', 'Phone']):
                    self.lead_state = LeadCaptureState.INFO_COMPLETE
                    logger.info("All lead information collected")


    def _get_missing_fields(self) -> List[str]:
        """Get list of missing required fields in lead information."""
        required_fields = ['Name', 'Email', 'Phone']
        return [field for field in required_fields 
                if field not in self.partial_lead_info or self.partial_lead_info[field] == "N/A"]


    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionaries."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    

    def _generate_response(self, message: str) -> str:
        try:
            print('Generate Response is calling before calling fetch and save aws data')
            self.fetch_and_save_aws_data()
            print('Generate Response is calling After calling fetch and save aws data')
            local_path = 'C:/Users/admin/Documents/Document/Bot/src/storeJsonRecord'
            aws_data_text = ""
            if os.path.exists(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                aws_data_text = json.dumps(data, indent=2)
            else:
                aws_data_text = "No AWS data available."
            
            aws_data_text = aws_data_text[:2000]
            context = self.get_combined_context(message)

            recent_messages = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
            topics_prompt = """Given these conversation messages, identify the main topic being discussed:\n{}\nReturn ONLY the topic being discussed, nothing else.""".format("\n".join(recent_messages))
            topic_response = self.llm.invoke(topics_prompt)
            current_topic = topic_response.content

            recent_messages_text = "\n".join(recent_messages)
            recent_messages_text = recent_messages_text[-1000:]

            doc_context = self._get_relevant_context(message)
            if len(doc_context) > 2000:
                doc_context = doc_context[:2000] + "..."

            system_context = f"""Current topic of discussion: {current_topic}
            Previous conversation context: {recent_messages_text}
            Product information: {doc_context}
            Lead information: {json.dumps(self.partial_lead_info, indent=2) if self.partial_lead_info else "No lead information yet"}
            Lead state: {self.lead_state.value}"""

            prompt = f"""You are an assistant for Emaar Properties sales team. Follow these rules STRICTLY:
 
            1. ONLY answer questions using information from the **System Context**.
            2. NEVER make up information. No assumptions or general answers are allowed.
            3. Keep responses clear, friendly, and human-like.
            4. Stay on topic: {current_topic}
            5. Reference past messages if relevant.
            6. DO NOT ask for user contact info unless they show real interest.
            7. Do not mention that you're using a PDF or system context unless asked.
            8. If the context is unrelated to the question, say you don’t have enough info.
            9. Keep your tone professional and concise.
            10. Show the AWS record based on the exact requirment.
            11.If **no relevant data** is found for the user's question **and the question is NOT about scheduling or connecting**, respond:
            - “I’m sorry, I don’t have enough information about that at the moment.”

            12. If the user says anything about **scheduling a meeting** or **connecting with someone**, do **NOT** show the fallback message. Instead, respond:
            - “Sure! Could you please share your Name, Email, and Phone number so I can schedule a meeting?”

            13. Treat the following as clear signals to schedule or connect (case-insensitive):
            - "schedule a meeting"
            - "book a call"
            - "connect me"
            - "talk to someone"
            - "speak to an agent"
            - "sales call"
            - "connect with a sales rep"
            - "want to discuss"
            14. If the **System Context** does not contain relevant data but exception of Schedule meeting and connect, reply: **"I’m sorry, I don’t have enough information about that at the moment."**
            
            System Context:
            {system_context}

            Human: {message}

            Assistant:"""
            print(f"system_context  :{system_context}")
            try:
                print("Before llm.invoke")
                response = self.llm.invoke(prompt)
                print("After llm.invoke")
                print("Model Response:", response.content)
                return response.content
            except Exception as e:
                print(f"Error during llm.invoke: {e}")
                return "Error calling language model."           
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Let's try again."
        

    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and handle lead capture if needed."""
        data_response = self.process_user_query(message)
        if data_response:
            print("\nBot:",data_response)
            self.conversation_history.append(f"Human: {message}")
            self.conversation_history.append(f"Assistant: {data_response['response']}")
            print("data_response:",data_response)
            return data_response
        """Process a user message and handle lead capture if needed."""
        try:
            self._update_lead_state(message)
            self.conversation_history.append(f"Human: {message}")
            response = self._generate_response(message)
            self.conversation_history.append(f"Assistant: {response}")
            self.conversation_history = self.conversation_history[-30:]
            logger.info(f"Conversation history: {self.conversation_history}")
            if self.lead_state == LeadCaptureState.INTEREST_DETECTED:
                missing_fields = self._get_missing_fields()
                if missing_fields:
                    if len(missing_fields) == 1:
                        response += f"\n\nCould you share your {missing_fields[0]}?"
                    else:
                        response += f"\n\nCould you share your {', '.join(missing_fields[:-1])} and {missing_fields[-1]}?"
            
            elif self.lead_state == LeadCaptureState.COLLECTING_INFO:
                missing_fields = self._get_missing_fields()
                if missing_fields:
                    if len(missing_fields) == 1:
                        response += f"\n\nJust need your {missing_fields[0]} to get started."
                    else:
                        response += f"\n\nJust need your {', '.join(missing_fields[:-1])} and {missing_fields[-1]} to get started."
                else:
                    self.lead_state = LeadCaptureState.INFO_COMPLETE
            
            elif self.lead_state == LeadCaptureState.INFO_COMPLETE:
                lead_created, lead_id = self.salesforce.create_lead(self.partial_lead_info)
                if lead_created:
                    self.current_lead_id = lead_id
                    logger.info("Lead information saved to Salesforce successfully")
                    self.lead_state = LeadCaptureState.AWAITING_MEETING_CONFIRMATION
                    response += "\n\nGreat! I've saved your information.\n\nDo you want to schedule meeting with FSTC Team Member? (Yes/No)"
                else:
                    logger.error("Failed to save lead information to Salesforce")
                    response += "\n\nSorry, I had trouble saving your information. Would you mind trying again?"

            elif self.lead_state == LeadCaptureState.AWAITING_MEETING_CONFIRMATION:
                if message.strip().lower() in ["yes", "yeah", "y", "sure", "please","schedule","schedule meeting"]:
                    self.available_slots = self.salesforce.show_availableMeeting() or []
                    if self.available_slots:
                        self.lead_state = LeadCaptureState.WAITING_MEETING_SLOT_SELECTION
                        slots_text = self.format_slots_nicely(self.available_slots)
                        response += f"\n\nHere are the available meeting slots for today: {slots_text}"
                    else:
                        self.lead_state = LeadCaptureState.NO_INTEREST
                        response += "\n\nSorry, I couldn’t fetch available meeting slots right now."
                else:
                    self.lead_state = LeadCaptureState.NO_INTEREST
                    response += "\n\nNo problem! Let me know if you have any other questions."

            elif self.lead_state == LeadCaptureState.WAITING_MEETING_SLOT_SELECTION:
                parsed_time = message.strip().lower()
                parsed_time = parsed_time.replace("\"", "").replace("'", "").replace(" ", "").replace(".", "")

                if parsed_time.isdigit():
                    if len(parsed_time) <= 2:
                        parsed_time = parsed_time.zfill(2) + ":00"     
                    elif len(parsed_time) == 3:
                        parsed_time = "0" + parsed_time[0] + ":" + parsed_time[1:]  
                    elif len(parsed_time) == 4:
                        parsed_time = parsed_time[:2] + ":" + parsed_time[2:]       
                elif ":" in parsed_time:
                    parts = parsed_time.split(":")
                    if len(parts) == 2 and all(p.isdigit() for p in parts):
                        parsed_time = parts[0].zfill(2) + ":" + parts[1].zfill(2)

                logger.info(f"✅ Final normalized time: {parsed_time}")
                logger.info(f"🔎 Comparing against available slots: {self.available_slots}")

                if parsed_time in self.available_slots and self.current_lead_id:
                    success = self.salesforce.create_meeting(self.current_lead_id, parsed_time)
                    self.lead_state = LeadCaptureState.NO_INTEREST
                    self.available_slots = []
                    self.current_lead_id = None
                    if success:
                        response = f"✅ Your meeting has been scheduled at {parsed_time}. Our team will contact you soon!"
                    else:
                        response = f"❌ Something went wrong while scheduling your meeting at {parsed_time}. Please try again."
                else:
                    response = f"⚠️ \"{message}\" is not a valid time. Please choose from: {', '.join(self.available_slots)}"


            return {
                "response": response,
                "lead_info": self.partial_lead_info if self.partial_lead_info else None,
                "lead_state": self.lead_state.value
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "Sorry, I encountered an error. Let's try again.",
                "lead_info": None,
                "lead_state": self.lead_state.value
            }


    def format_slots_nicely(self, slots: List[str], columns: int = 3) -> str:
        """Formats meeting slots in a professional table layout"""
        if not slots:
            return "No available time slots at the moment."
        max_length = max(len(slot) for slot in slots) 
        column_width = max_length + 5  
        rows = []
        for i in range(0, len(slots), columns):
            row = []
            for j in range(columns):
                idx = i + j
                if idx < len(slots):
                    row.append(f"{slots[idx]:>{column_width}}")
            rows.append("".join(row)) 
        
        return (
             "Available meeting times:\n\n" +
            "\n".join(rows) +  
            "\n\nPlease pick one."
        )


    def interpret_user_query(self, user_question: str) -> Optional[Dict[str, Any]]:
        prompt = f"""
        Analyze this query and extract filtering criteria in JSON format with:
        - Show the aws record in aws when creteria match
        - field: exact field name to filter
        - condition: equals/contains/greater_than/less_than/greater_than_or_equal/less_than_or_equal
        - value: value to compare
        - sort: optional, either "asc" (for cheapest) or "desc" (for costliest)
        
        Supported fields: Name, Area, Area_Code, Plot_Number, 
                        Project_Name, Project_Type, Building_Number,
                        Tower_Name, Unit_No, Unit_Type, Unit_Price, District
        
        Examples:
        Input: "Show properties under or equal to 2000"
        Output: {{"field": "Unit_Price", "condition": "less_than_or_equal", "value": "2000"}}

        Input: "Find properties 2000 or above"
        Output: {{"field": "Unit_Price", "condition": "greater_than_or_equal", "value": "2000"}}
        
        Input: "Show me the cheapest properties"
        Output: {{"field": "Unit_Price", "sort": "asc"}}
        
        Input: "Show me the costliest properties"
        Output: {{"field": "Unit_Price", "sort": "desc"}}
        
        Input: "Show me cheapest studio apartments"
        Output: [{{"field": "Unit_Type", "condition": "equals", "value": "Studio Apartment"}},
                {{"field": "Unit_Price", "sort": "asc"}}]
        
        Input: "{user_question}"
        Output:"""
        
        try:
            response = self.llm.invoke(prompt)
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Error interpreting query: {e}")
            return None    


    def filter_data_by_criteria(self, data: List[Any], criteria: Union[Dict, List[Dict]], user_question: str = "") -> List[Any]:
        if not criteria:
            return data
            
        # Convert single criteria to list for uniform handling
        if isinstance(criteria, dict):
            criteria_list = [criteria]
        else:
            criteria_list = criteria
            
        filtered_data = data
        
        # First apply all filters
        for criteria in criteria_list:
            if not criteria:
                continue
                
            field = criteria.get('field', '').lower().replace(' ', '_')
            condition = criteria.get('condition', '').lower()
            value = criteria.get('value', '').strip()
            sort_order = criteria.get('sort', '').lower()
            
            if not field:
                continue
                
            # Handle sorting separately
            if sort_order and field:
                continue
                
            current_filtered = []
            
            for item in filtered_data:
                record = item.dict() if isinstance(item, Document) else item
                content = record.get('page_content', '')
                
                # Extract all fields from content
                fields_data = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, val = line.split(':', 1)
                        normalized_key = key.strip().lower().replace(' ', '_')
                        fields_data[normalized_key] = val.strip()
                
                if field not in fields_data:
                    continue
                    
                record_value = fields_data[field]
                
                try:
                    # Numeric fields
                    if field in ['plot_number', 'unit_price', 'unit_no', 'area_code', 'building_number']:
                        try:
                            record_num = float(record_value.replace(',', '').replace('₹', ''))
                            filter_num = float(value.replace(',', '').replace('₹', ''))
                            
                            if condition == 'equals' and record_num == filter_num:
                                current_filtered.append(item)
                            elif condition == 'greater_than' and record_num > filter_num:
                                current_filtered.append(item)
                            elif condition == 'less_than' and record_num < filter_num:
                                current_filtered.append(item)
                            elif condition == 'greater_than_or_equal' and record_num >= filter_num:
                                current_filtered.append(item)
                            elif condition == 'less_than_or_equal' and record_num <= filter_num:
                                current_filtered.append(item)
                            elif condition == 'contains' and str(filter_num) in str(record_num):
                                current_filtered.append(item)
                        except ValueError:
                            continue
                    
                    # Text fields
                    else:
                        record_text = str(record_value).lower()
                        filter_text = str(value).lower()
                        
                        if condition == 'equals' and record_text == filter_text:
                            current_filtered.append(item)
                        elif condition == 'contains' and filter_text in record_text:
                            current_filtered.append(item)
                        elif condition == 'starts_with' and record_text.startswith(filter_text):
                            current_filtered.append(item)
                        elif condition == 'ends_with' and record_text.endswith(filter_text):
                            current_filtered.append(item)
                            
                except Exception as e:
                    logger.warning(f"Filtering error for field {field}: {e}")
                    continue
                    
            filtered_data = current_filtered
        
        # Then apply sorting if specified
        for criteria in criteria_list:
            if not criteria:
                continue
                
            field = criteria.get('field', '').lower().replace(' ', '_')
            sort_order = criteria.get('sort', '').lower()
            
            if field and sort_order:
                try:
                    # Create a list of tuples (sort_key, item) for sorting
                    sortable = []
                    for item in filtered_data:
                        record = item.dict() if isinstance(item, Document) else item
                        content = record.get('page_content', '')
                        
                        fields_data = {}
                        for line in content.split('\n'):
                            if ':' in line:
                                key, val = line.split(':', 1)
                                normalized_key = key.strip().lower().replace(' ', '_')
                                fields_data[normalized_key] = val.strip()
                        
                        if field in fields_data:
                            try:
                                if field in ['plot_number', 'unit_price', 'unit_no', 'area_code', 'building_number']:
                                    sort_key = float(fields_data[field].replace(',', '').replace('₹', ''))
                                else:
                                    sort_key = fields_data[field].lower()
                                sortable.append((sort_key, item))
                            except ValueError:
                                continue
                    
                    # Sort the data
                    if sort_order == 'asc':
                        sortable.sort(key=lambda x: x[0])
                    elif sort_order == 'desc':
                        sortable.sort(key=lambda x: x[0], reverse=True)
                    
                    # Extract just the items in sorted order
                    filtered_data = [item for (sort_key, item) in sortable]
                    
                    # For cheapest/costliest, we might want just the top result
                    if user_question and ("cheapest" in user_question.lower() or "costliest" in user_question.lower()):
                        filtered_data = filtered_data[:1]
                        
                except Exception as e:
                    logger.error(f"Error sorting data: {e}")
                    continue
                    
        return filtered_data


    def fetch_and_filter_data(self, user_question: str) -> List[Dict[str, Any]]:
        criteria = self.interpret_user_query(user_question)
        if not criteria:
            try:
                lower_question = user_question.lower()           
                if ("show all" in lower_question or "all record" in lower_question or "all aws record" in lower_question or "give me all" in lower_question):
                    try:
                        data = self.fetch_and_save_aws_data()
                        if not data:
                            return [{"message": "No data available"}]
                        return [doc.dict() if isinstance(doc, Document) else doc for doc in data]
                    except Exception as e:
                        logger.error(f"Error fetching all data: {e}")
                        return [{"message": "Error processing your request"}]
                else:
                    return [{"message": "No data available"}]
            except Exception as e:
                logger.error(f"Error fetching all data: {e}")
                return [{"message": "Error processing your request"}]
        
        try:
            data = self.fetch_and_save_aws_data()
            if not data:
                return [{"message": "No data available"}]
                
            filtered = self.filter_data_by_criteria(data, criteria)
            if not filtered:
                return [{"message": "No data available."}]
                
            return [doc.dict() if isinstance(doc, Document) else doc for doc in filtered]
            
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return [{"message": "Error processing your request"}]


    def process_user_query(self, message: str) -> Optional[Dict[str, Any]]:
        command_keywords = ["show", "find", "search", "filter", "get", "list", "tell me", "what is"]
        if any(keyword in message.lower() for keyword in command_keywords):
            results = self.fetch_and_filter_data(message)
            
            if not results:
                return {
                    "response": "No data available.",
                    "lead_info": self.partial_lead_info,
                    "lead_state": self.lead_state.value
                }
                
            if len(results) == 1 and "message" in results[0]:
                return {
                    "response": results[0]["message"],
                    "lead_info": self.partial_lead_info,
                    "lead_state": self.lead_state.value
                }
            
            table_data = []
            headers = [
                "Name", "Area", "Plot #", "Project", 
                "Unit Type", "Price", "District"
            ]
            
            for record in results:
                if isinstance(record, dict):
                    content = record.get('page_content', '')
                    fields = {}
                    for line in content.split('\n'):
                        if ':' in line:
                            key, val = line.split(':', 1)
                            fields[key.strip()] = val.strip()
                    
                    table_data.append([
                        fields.get('Name', 'N/A'),
                        fields.get('Area', 'N/A'),
                        fields.get('Plot Number', 'N/A'),
                        fields.get('Project Name', 'N/A'),
                        fields.get('Unit Type', 'N/A'),
                        fields.get('Unit Price', 'N/A'),
                        fields.get('District', 'N/A')
                    ])
            
            if not table_data:
                response_text = "No matching records found."
            else:
                from tabulate import tabulate
                table_str = tabulate(
                    table_data, 
                    headers=headers,
                    tablefmt="grid",
                    numalign="right",
                    stralign="left"
                )
                total_records = len(table_data)
                response_text = f"```\n{table_str}\n```" 
                response_text += f"\n\nTotal Records found: {total_records}"
                
                if "cheapest" in message.lower() or "costliest" in message.lower():
                    if table_data:
                        price = table_data[0][5]  
                        response_text = f"The {'cheapest' if 'cheapest' in message.lower() else 'costliest'} property is:\n\n{response_text}"
            
            return {
                "response": response_text,
                "lead_info": self.partial_lead_info,
                "lead_state": self.lead_state.value
            }
        return None


def main():
    """Main function to run the sales RAG chatbot."""
    try:
        # pdf_path = 'C:/Users/admin/Documents/Document/PropFinder/Emaar_FAQ.pdf' 
        pdf_path = '/home/ubuntu/AgenticBotImplementationAWS/Emaar_FAQ.pdf'
        chatbot = SalesRAGBot(pdf_path)
        
        print("Welcome to the Sales Assistant!")
        print("I can help you learn more about our products and services.")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! Thank you for your interest.")
                break
                
            if not user_input:
                print("Please enter a message.")
                continue
            
            print("Before printing response")
            response = chatbot.process_message(user_input)
            print("\nBot:", response['response'])
        
            if response['lead_info']:
                print("\n[Lead information captured:", response['lead_info'], "]")
                print("[Current state:", response['lead_state'], "]")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
