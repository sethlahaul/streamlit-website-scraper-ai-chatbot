import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Website Parser and Chatbot with Gemini AI
class GeminiWebsiteChatbot:
    def __init__(self, api_key: str = None):
        self.content = ""
        self.website_data = {}
        self.api_key = api_key
        self.model = None
        
        if api_key:
            self.setup_gemini(api_key)
    
    def setup_gemini(self, api_key: str):
        """Setup Gemini AI model"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.api_key = api_key
            return True
        except Exception as e:
            st.error(f"Failed to setup Gemini API: {str(e)}")
            return False
    
    def parse_website(self, url: str) -> Dict:
        """Parse website content and extract text"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '').strip() if meta_desc else ""
            
            # Extract headings
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_text = h.get_text().strip()
                if heading_text and len(heading_text) < 200:  # Reasonable heading length
                    headings.append(heading_text)
            
            # Extract main content paragraphs
            paragraphs = []
            for p in soup.find_all('p'):
                para_text = p.get_text().strip()
                if len(para_text) > 50:  # Only substantial paragraphs
                    paragraphs.append(para_text)
            
            # Limit content size for API efficiency (Gemini has context limits)
            max_content_length = 50000  # Adjust based on needs
            if len(text) > max_content_length:
                text = text[:max_content_length] + "... [Content truncated]"
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'headings': headings[:20],  # Limit headings
                'paragraphs': paragraphs[:30],  # Limit paragraphs
                'content': text,
                'word_count': len(text.split()),
                'char_count': len(text),
                'status': 'success'
            }
            
        except requests.RequestException as e:
            return {
                'url': url,
                'error': f"Request failed: {str(e)}",
                'status': 'error'
            }
        except Exception as e:
            return {
                'url': url,
                'error': f"Parsing failed: {str(e)}",
                'status': 'error'
            }
    
    def process_content(self, website_data: Dict):
        """Process and store website content"""
        self.website_data = website_data
        self.content = website_data['content']
    
    def generate_response(self, query: str) -> str:
        """Generate AI response using Gemini"""
        if not self.model:
            return "❌ Gemini API is not configured. Please add your API key."
        
        if not self.content:
            return "❌ No website content available. Please parse a website first!"
        
        try:
            # Create context-aware prompt
            prompt = f"""
You are an AI assistant that helps users understand and get information from website content. 

Website Information:
- Title: {self.website_data.get('title', 'N/A')}
- URL: {self.website_data.get('url', 'N/A')}
- Description: {self.website_data.get('description', 'N/A')}

Main headings from the website:
{chr(10).join(['• ' + h for h in self.website_data.get('headings', [])[:10]])}

Website Content:
{self.content[:10000]}  

User Question: {query}

Please provide a helpful, accurate answer based on the website content above. If the information isn't available in the content, say so clearly. Be conversational and helpful.
"""
            
            # Configure safety settings to be less restrictive for general content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper():
                return "❌ Invalid API key. Please check your Gemini API key."
            elif "SAFETY" in error_msg.upper():
                return "⚠️ Response blocked by safety filters. Try rephrasing your question."
            elif "QUOTA" in error_msg.upper():
                return "❌ API quota exceeded. Please try again later or check your Gemini API usage."
            else:
                return f"❌ Error generating response: {error_msg}"

def main():
    st.set_page_config(
        page_title="AI Website Parser & Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🌐 AI Website Parser & Chatbot")
    st.markdown("Parse any website and chat with its content using Google's Gemini AI!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = GeminiWebsiteChatbot()
    if 'parsed_data' not in st.session_state:
        st.session_state.parsed_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_configured' not in st.session_state:
        st.session_state.api_configured = False
    
    # Sidebar for configuration and website parsing
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Gemini API Key input
        api_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Get your free API key from https://makersuite.google.com/app/apikey",
            placeholder="Enter your Gemini API key..."
        )
        
        if api_key and not st.session_state.api_configured:
            if st.session_state.chatbot.setup_gemini(api_key):
                st.session_state.api_configured = True
                st.success("✅ Gemini API configured successfully!")
            else:
                st.error("❌ Failed to configure Gemini API")
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key to use AI features")
            st.markdown("**Get your free API key:**")
            st.markdown("1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)")
            st.markdown("2. Click 'Create API Key'")
            st.markdown("3. Copy and paste it above")
        
        st.divider()
        st.header("🔍 Website Parser")
        
        url_input = st.text_input(
            "Enter Website URL:",
            placeholder="https://example.com",
            help="Enter the full URL including https://"
        )
        
        if st.button("Parse Website", type="primary", disabled=not url_input):
            with st.spinner("Parsing website..."):
                parsed_data = st.session_state.chatbot.parse_website(url_input)
                st.session_state.parsed_data = parsed_data
                
                if parsed_data['status'] == 'success':
                    st.session_state.chatbot.process_content(parsed_data)
                    st.success("✅ Website parsed successfully!")
                    
                    # Clear previous chat when new website is parsed
                    st.session_state.chat_history = []
                else:
                    st.error(f"❌ Error: {parsed_data.get('error', 'Unknown error')}")
        
        # Show parsed website info
        if st.session_state.parsed_data and st.session_state.parsed_data['status'] == 'success':
            st.subheader("📄 Parsed Content Info")
            data = st.session_state.parsed_data
            st.write(f"**Title:** {data['title'][:50]}{'...' if len(data['title']) > 50 else ''}")
            st.write(f"**Word Count:** {data['word_count']:,}")
            st.write(f"**Character Count:** {data['char_count']:,}")
            
            if data['headings']:
                st.write("**Main Headings:**")
                for heading in data['headings'][:5]:
                    st.write(f"• {heading[:60]}{'...' if len(heading) > 60 else ''}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with Website Content")
        
        if not st.session_state.api_configured:
            st.info("🔑 Please configure your Gemini API key in the sidebar to start chatting!")
        elif not st.session_state.parsed_data or st.session_state.parsed_data['status'] != 'success':
            st.info("🌐 Please parse a website using the sidebar to start chatting!")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)
        
        # Chat input
        if prompt := st.chat_input("Ask anything about the website content..."):
            if st.session_state.api_configured and st.session_state.parsed_data and st.session_state.parsed_data['status'] == 'success':
                # Add user message to chat
                st.session_state.chat_history.append(("user", prompt))
                st.chat_message("user").write(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("🤖 AI is thinking..."):
                        response = st.session_state.chatbot.generate_response(prompt)
                        st.write(response)
                        st.session_state.chat_history.append(("assistant", response))
            elif not st.session_state.api_configured:
                st.warning("⚠️ Please configure your Gemini API key first!")
            else:
                st.warning("⚠️ Please parse a website first!")
    
    with col2:
        st.header("📊 Content Analysis")
        
        if st.session_state.parsed_data and st.session_state.parsed_data['status'] == 'success':
            data = st.session_state.parsed_data
            
            # Basic stats
            st.metric("Total Words", f"{data['word_count']:,}")
            st.metric("Characters", f"{data['char_count']:,}")
            st.metric("Headings Found", len(data['headings']))
            st.metric("Paragraphs", len(data.get('paragraphs', [])))
            
            # Website info
            st.subheader("🌐 Website Info")
            st.write(f"**URL:** {data['url']}")
            if data['description']:
                st.write(f"**Description:** {data['description'][:100]}{'...' if len(data['description']) > 100 else ''}")
            
            # Sample content
            st.subheader("📝 Sample Content")
            if data.get('paragraphs'):
                sample_text = data['paragraphs'][0] if data['paragraphs'] else ""
                st.write(sample_text[:300] + "..." if len(sample_text) > 300 else sample_text)
        else:
            st.info("Parse a website to see content analysis")
    
    # Action buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col_b:
        if st.button("🔄 Reset All"):
            st.session_state.chat_history = []
            st.session_state.parsed_data = None
            st.session_state.api_configured = False
            st.session_state.chatbot = GeminiWebsiteChatbot()
            st.rerun()
    
    # Instructions and tips
    with st.expander("ℹ️ How to Use & Tips"):
        st.markdown("""
        ## 🚀 Getting Started
        1. **Get Gemini API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) and create a free API key
        2. **Configure API**: Enter your API key in the sidebar
        3. **Parse Website**: Enter any website URL and click "Parse Website"
        4. **Start Chatting**: Ask questions about the website content using natural language
        
        ## 💡 Sample Questions to Try
        - "What is this website about?"
        - "Summarize the main points"
        - "Tell me about [specific topic mentioned on the site]"
        - "What are the key features mentioned?"
        - "Who is this website for?"
        - "What services/products are offered?"
        
        ## 🔧 Features
        - **AI-Powered**: Uses Google's Gemini AI for intelligent responses
        - **Smart Parsing**: Extracts clean content from any website
        - **Context Aware**: AI understands the website structure and content
        - **Free to Use**: Uses Gemini's free tier (with usage limits)
        
        ## ⚠️ Notes
        - Free Gemini API has usage limits (check Google AI Studio for details)
        - Some websites may block automated parsing
        - Large websites are automatically truncated for efficiency
        """)
    
    # Footer
    st.divider()
    st.markdown("**Powered by Google Gemini AI** | Built with Streamlit")

if __name__ == "__main__":
    main()