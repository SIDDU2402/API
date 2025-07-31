from fastapi import FastAPI, HTTPException
import json
import requests
import tempfile
import os
from typing import List, Dict, Any
import google.generativeai as genai

# Import models first
from models import Clause, QueryRequest, QueryResponse
from pydantic import BaseModel

# Import services after models
from services import gemini_service, redis_service, pinecone_service
from config import config
from document_uploader import DocumentProcessor

app = FastAPI(title="Document Query Assistant", version="1.0.0")

async def generate_hackrx_answer(question: str, clauses: List[Clause]) -> str:
    """Generate concise, direct answer for HackRX format"""
    
    # Prepare clauses context
    clauses_context = "\n".join([
        f"Clause {i+1}: {clause.content}"
        for i, clause in enumerate(clauses)
    ])
    
    prompt = f"""You are an expert insurance policy analyst. Based on the provided policy clauses, answer the question directly and concisely.

Policy Clauses:
{clauses_context}

Question: "{question}"

Instructions:
1. Provide a direct, factual answer based ONLY on the provided clauses
2. Be specific about numbers, timeframes, percentages, and conditions
3. If the information is not in the clauses, say "Information not available in the provided policy document"
4. Keep the answer concise but complete
5. Do not add explanations or interpretations beyond what's stated
6. Return ONLY the answer text, no JSON formatting

Answer:"""

    try:
        # Generate response using Gemini
        response = gemini_service.model.generate_content(prompt)
        answer = response.text.strip()
        
        # Clean up any unwanted formatting
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        return answer
        
    except Exception as e:
        print(f"❌ Error generating HackRX answer: {e}")
        return f"Error processing question: {str(e)}"

# Hackathon-specific models
class HackRXRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]  # List of questions to ask

class HackRXResponse(BaseModel):
    answers: List[str]  # List of direct answers

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process document query and return structured response"""
    try:
        # Check if Gemini service is available
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        
        # Generate cache key if Redis is available
        cached_response = None
        if redis_service:
            clause_ids = [clause.clause_id for clause in request.top_k_clauses]
            query_hash = redis_service.generate_query_hash(request.user_question, clause_ids)
            cached_response = redis_service.get_cached_response(query_hash)
        
        # Return cached response if available
        if cached_response:
            print("✅ Returning cached response")
            return cached_response
        
        # Process with Gemini if not cached
        response = gemini_service.analyze_clauses(request.user_question, request.top_k_clauses)
        
        # Cache the response if Redis is available
        if redis_service:
            redis_service.cache_response(query_hash, response)
            print("✅ Response cached for future requests")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document Query Assistant"}

@app.get("/test-redis")
async def test_redis():
    """Test Redis connection endpoint"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    result = redis_service.test_connection()
    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@app.get("/cache/stats")
async def get_cache_stats():
    """Get document cache statistics"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    try:
        stats = redis_service.get_document_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache():
    """Clear document cache"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    try:
        success = redis_service.clear_document_cache()
        if success:
            return {"status": "success", "message": "Document cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear document cache")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """HackRX endpoint - Process document from URL and answer multiple questions"""
    try:
        print(f"🚀 HackRX request received with {len(request.questions)} questions")
        
        # Check services availability
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        if not pinecone_service or not pinecone_service.index:
            raise HTTPException(status_code=503, detail="Pinecone service not available")
        
        # Check if document is already cached in Redis
        document_name = "hackrx_policy"
        
        if redis_service and redis_service.is_document_cached(request.documents):
            print("🚀 Document found in Redis cache - using cached version!")
            cached_info = redis_service.get_cached_document_info(request.documents)
            print(f"📋 Cached document: {cached_info['document_name']} ({cached_info['clause_count']} clauses)")
            
            # Check if embeddings are uploaded to Pinecone
            if not redis_service.are_embeddings_uploaded(request.documents):
                print("🔄 Embeddings not in Pinecone - uploading from Redis cache...")
                cached_clauses = redis_service.get_cached_clauses(request.documents)
                
                processor = DocumentProcessor()
                success = processor.upload_to_pinecone(cached_clauses)
                if success:
                    redis_service.mark_embeddings_uploaded(request.documents)
                    print("✅ Cached clauses uploaded to Pinecone")
                else:
                    raise HTTPException(status_code=500, detail="Failed to upload cached clauses to vector database")
            else:
                print("✅ Embeddings already in Pinecone - ready to query!")
        
        else:
            print("📥 Document not cached - downloading and processing...")
            print(f"🔗 URL: {request.documents}")
            
            # Download the PDF
            response = requests.get(request.documents, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Process the document
                print("🔄 Processing document...")
                processor = DocumentProcessor()
                
                # Extract text and create clauses
                text = processor.extract_text_from_file(temp_file_path)
                clauses = processor.split_document_into_clauses(text, document_name)
                
                print(f"✅ Document processed into {len(clauses)} clauses")
                
                # Cache the processed document in Redis
                if redis_service:
                    file_size = len(response.content)
                    cache_success = redis_service.cache_document(
                        request.documents, 
                        document_name, 
                        clauses, 
                        file_size, 
                        text
                    )
                    
                    if cache_success:
                        print("✅ Document cached in Redis for future requests")
                    else:
                        print("⚠️ Failed to cache document in Redis (continuing anyway)")
                
                # Upload to Pinecone
                success = processor.upload_to_pinecone(clauses)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to upload document to vector database")
                
                # Mark embeddings as uploaded in Redis
                if redis_service:
                    redis_service.mark_embeddings_uploaded(request.documents)
                print("✅ Document uploaded to vector database")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Process questions in parallel for maximum speed
        import asyncio
        
        async def process_single_question(question: str, question_index: int) -> str:
            """Process a single question asynchronously"""
            print(f"🔍 Processing question {question_index}/{len(request.questions)}: {question[:50]}...")
            
            try:
                # Check semantic cache first for super fast responses
                if redis_service:
                    cached_answer = redis_service.get_semantic_cached_response(question, document_name)
                    if cached_answer:
                        print(f"⚡ Using semantic cached answer for question {question_index}")
                        return cached_answer
                
                # Generate embedding for the question
                genai.configure(api_key=config.GEMINI_API_KEY)
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=question,
                    task_type="retrieval_query"
                )
                query_embedding = result['embedding']
                
                # Search for relevant clauses with improved parameters
                search_results = pinecone_service.index.query(
                    vector=query_embedding,
                    top_k=8,  # Increased for better context
                    include_metadata=True,
                    filter={"document": document_name}
                )
                
                if not search_results.matches:
                    return "No relevant information found in the document"
                
                # Convert to Clause objects
                relevant_clauses = []
                for match in search_results.matches:
                    clause = Clause(
                        clause_id=match.id,
                        content=match.metadata.get('content', ''),
                        score=match.score
                    )
                    relevant_clauses.append(clause)
                
                # Generate concise answer using specialized HackRX prompt
                answer = await generate_hackrx_answer(question, relevant_clauses)
                
                # Cache the answer for semantic similarity
                if redis_service:
                    redis_service.cache_semantic_response(question, document_name, answer)
                
                print(f"✅ Question {question_index} processed successfully")
                return answer
                
            except Exception as e:
                print(f"❌ Error processing question {question_index}: {e}")
                return f"Error processing question: {str(e)}"
        
        # Process all questions concurrently (limit to 3 concurrent to avoid rate limits)
        semaphore = asyncio.Semaphore(3)
        
        async def process_with_semaphore(question: str, index: int) -> str:
            async with semaphore:
                return await process_single_question(question, index + 1)
        
        # Create tasks for all questions
        tasks = [
            process_with_semaphore(question, i) 
            for i, question in enumerate(request.questions)
        ]
        
        # Execute all tasks concurrently
        print(f"🚀 Processing {len(request.questions)} questions concurrently...")
        answers = await asyncio.gather(*tasks)
        
        print(f"🎉 All {len(request.questions)} questions processed!")
        return HackRXResponse(answers=answers)
                
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        print(f"❌ HackRX processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)