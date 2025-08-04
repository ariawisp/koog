package ai.koog.agents.memory.retrieval.providers

import ai.koog.agents.memory.model.Fact
import ai.koog.agents.memory.model.MemoryScope
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.agents.memory.providers.AgentMemoryProvider
import ai.koog.agents.memory.retrieval.Provenance
import ai.koog.agents.memory.retrieval.RetrievalProvider
import ai.koog.agents.memory.retrieval.RetrievalQuery
import ai.koog.agents.memory.retrieval.RetrievalRecipe
import ai.koog.agents.memory.retrieval.RetrievalResult
import ai.koog.agents.memory.retrieval.RetrievalTarget
import ai.koog.rag.base.RankedDocumentStorage
import ai.koog.rag.base.mostRelevantDocuments

/**
 * Default retrieval provider that works with existing vector-based document storage
 * and memory providers. This serves as a bridge to the existing RAG functionality.
 */
public class VectorRetrievalProvider(
    private val memoryProvider: AgentMemoryProvider? = null,
    private val documentStorage: RankedDocumentStorage<String>? = null
) : RetrievalProvider {
    
    override fun supports(recipe: RetrievalRecipe): Boolean = when(recipe) {
        RetrievalRecipe.VECTOR_SIMILARITY, 
        RetrievalRecipe.TEXT_BM25,
        RetrievalRecipe.HYBRID_RRF -> true
        else -> false
    }
    
    override suspend fun retrieve(query: RetrievalQuery): List<RetrievalResult> {
        val results = mutableListOf<RetrievalResult>()
        
        // Retrieve from memory facts if available
        if (memoryProvider != null && query.target in setOf(RetrievalTarget.FACTS, RetrievalTarget.ALL)) {
            results.addAll(retrieveFromMemory(query))
        }
        
        // Retrieve from document storage if available
        if (documentStorage != null && query.target in setOf(RetrievalTarget.DOCUMENTS, RetrievalTarget.ALL)) {
            results.addAll(retrieveFromDocuments(query))
        }
        
        // Sort by score and limit
        return results
            .sortedByDescending { it.score }
            .take(query.k)
    }
    
    private suspend fun retrieveFromMemory(query: RetrievalQuery): List<RetrievalResult> {
        val results = mutableListOf<RetrievalResult>()
        
        // Get subjects and scopes from filters or use defaults
        val subjects = query.filters.subjects.ifEmpty { 
            listOf(MemorySubject.Everything) 
        }
        val scopes = query.filters.scopes.ifEmpty { 
            listOf(MemoryScope.CrossProduct) 
        }
        
        // Search facts by description
        for (subject in subjects) {
            for (scope in scopes) {
                val facts = memoryProvider!!.loadByDescription(
                    description = query.text,
                    subject = subject,
                    scope = scope
                )
                
                // Convert facts to retrieval results
                facts.forEach { fact ->
                    results.add(factToResult(fact, subject, scope))
                }
            }
        }
        
        return results
    }
    
    private fun factToResult(fact: Fact, subject: MemorySubject, scope: MemoryScope): RetrievalResult {
        val content = when(fact) {
            is ai.koog.agents.memory.model.SingleFact -> 
                "${fact.concept.keyword}: ${fact.value}"
            is ai.koog.agents.memory.model.MultipleFacts -> 
                "${fact.concept.keyword}: ${fact.values.joinToString(", ")}"
        }
        
        return RetrievalResult(
            content = content,
            score = 0.8, // Default score for memory facts
            provenance = listOf(
                Provenance.Fact(
                    conceptKeyword = fact.concept.keyword,
                    subject = subject.name,
                    scope = scope.toString(),
                    timestamp = fact.timestamp
                )
            ),
            metadata = mapOf(
                "type" to "fact",
                "description" to fact.concept.description
            )
        )
    }
    
    private suspend fun retrieveFromDocuments(query: RetrievalQuery): List<RetrievalResult> {
        val docs = documentStorage!!.mostRelevantDocuments(
            query = query.text,
            count = query.k,
            similarityThreshold = 0.0
        )
        
        return docs.map { doc ->
            RetrievalResult(
                content = doc,
                score = 0.7, // Default score for documents
                provenance = listOf(
                    Provenance.Document(
                        path = "document",
                        lineStart = null,
                        lineEnd = null
                    )
                ),
                metadata = mapOf("type" to "document")
            )
        }
    }
}