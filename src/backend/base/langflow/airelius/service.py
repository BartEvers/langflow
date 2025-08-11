from __future__ import annotations

from typing import Any
import os
import glob
import json

from langflow.airelius.kernel import PFU_KERNEL
from langflow.airelius.retriever import Retriever
from langflow.airelius.summarize import build_flow_signature
from langflow.logging import logger


class PFUService:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def compose_prompt(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]):
        """Compose the full prompt for PFU planning."""
        parts = []
        
        # Add the core PFU kernel
        parts.append(PFU_KERNEL)
        
        # Add RAG context for component understanding - STRUCTURED and USEFUL
        if retrieved_snippets:
            # Include more relevant snippets and structure them better
            relevant_snippets = retrieved_snippets[:8]  # Increased to 8 for better coverage
            parts.append("\n" + "="*60)
            parts.append("RELEVANT COMPONENTS FOR THIS REQUEST:")
            parts.append("="*60)
            
            for i, snippet in enumerate(relevant_snippets):
                # Extract key information from snippet
                text = snippet.get('text', '')
                metadata = snippet.get('metadata', {})
                
                # Structure the component information clearly
                component_info = f"\n--- Component {i+1} ---\n"
                
                # Add metadata if available
                if metadata:
                    if 'component_type' in metadata:
                        component_info += f"Type: {metadata['component_type']}\n"
                    if 'display_name' in metadata:
                        component_info += f"Name: {metadata['display_name']}\n"
                    if 'description' in metadata:
                        component_info += f"Description: {metadata['description']}\n"
                
                # Add the actual code/text (truncated for readability)
                if len(text) > 300:
                    component_info += f"Code: {text[:300]}...\n"
                else:
                    component_info += f"Code: {text}\n"
                
                parts.append(component_info)
            
            parts.append("\n" + "="*60)
            parts.append("COMPONENT USAGE GUIDELINES:")
            parts.append("="*60)
            parts.append("""
IMPORTANT: Use these components as reference for:
1. CORRECT component structure and properties
2. Proper data field formats and types
3. Valid template configurations
4. Base classes and inheritance patterns
5. Input/output specifications

When creating operations, ensure they match the patterns shown in these components.
""")
        
        # Add current flow context - include full JSON for AI analysis
        if flow_data:
            import json
            try:
                # Include the full current flow JSON so AI can analyze the actual structure
                flow_json = json.dumps(flow_data, indent=2)
                parts.append("\n" + "="*60)
                parts.append("CURRENT FLOW STRUCTURE (FULL JSON):")
                parts.append("="*60)
                parts.append(flow_json)
                parts.append("="*60)
                
                # Also provide a quick summary for context
                nodes = flow_data.get("nodes", [])
                edges = flow_data.get("edges", [])
                flow_summary = f"Current flow has {len(nodes)} nodes and {len(edges)} edges"
                if nodes:
                    node_types = [node.get("type", "unknown") for node in nodes[:5]]  # Show first 5 node types
                    flow_summary += f". Node types: {', '.join(node_types)}"
                    if len(nodes) > 5:
                        flow_summary += f" and {len(nodes) - 5} more"
                
                parts.append(f"\nFLOW SUMMARY: {flow_summary}")
            except Exception as e:
                logger.warning(f"[PFU] Failed to serialize flow data: {e}")
                # Fallback to summary only
                nodes = flow_data.get("nodes", [])
                edges = flow_data.get("edges", [])
                flow_summary = f"Current flow has {len(nodes)} nodes and {len(edges)} edges"
                parts.append(f"\nCURRENT FLOW:\n{flow_summary}")
        
        # Add user request - make it the focus
        parts.append("\n" + "="*60)
        parts.append("USER REQUEST:")
        parts.append("="*60)
        parts.append(user_prompt)
        
        # Add focused planning instructions with component-specific guidance
        parts.append("""
PLANNING INSTRUCTIONS:

1. Start with <inner_monologue> explaining your strategy
2. Focus on the user's specific request
3. Use the available components as TEMPLATES for your operations
4. Ensure your operations match the component patterns shown above
5. Keep operations simple and logical
6. Validate that component structures match the examples provided
7. Respond with inner_monologue and a valid JSON plan

COMPONENT INTEGRATION STRATEGY:
- Study the component examples above for structure patterns
- Use similar data field formats and template configurations
- Ensure base_classes and inheritance match the examples
- Follow the same input/output specification patterns
- Maintain consistency with existing component architecture

Generate the PFU plan now:""")
        
        return "\n".join(parts)

    def parse_plan_from_llm_response(self, llm_content: str) -> dict[str, Any]:
        """Parse a structured plan from LLM response with improved error handling.
        
        This method attempts multiple parsing strategies to extract a valid plan
        from the LLM response, with detailed logging and fallback handling.
        """
        import json
        import re
        
        logger.info(f"[PFU] Attempting to parse plan from LLM response ({len(llm_content)} characters)")
        logger.info(f"[PFU] Full LLM response: {llm_content}")
        
        # Strategy 1: Look for JSON wrapped in code blocks
        json_patterns = [
            # Look for JSON wrapped in ```json blocks
            r'```json\s*(\{.*?\})\s*```',
            # Look for JSON wrapped in ``` blocks
            r'```\s*(\{.*?\})\s*```',
            # Look for bare JSON (more restrictive to avoid false positives)
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
        
        json_str = None
        used_pattern = None
        
        for i, pattern in enumerate(json_patterns):
            match = re.search(pattern, llm_content, re.DOTALL)
            if match:
                json_str = match.group(1)
                used_pattern = f"pattern {i+1}: {pattern}"
                logger.info(f"[PFU] Found JSON using {used_pattern}")
                break
        
        if not json_str:
            logger.warning("[PFU] No JSON found in LLM response using standard patterns")
            logger.debug(f"[PFU] Full LLM response: {llm_content}")
            
            # Strategy 2: Try to find any JSON-like structure
            # Look for content between the last { and the last }
            last_open = llm_content.rfind('{')
            last_close = llm_content.rfind('}')
            
            if last_open != -1 and last_close != -1 and last_close > last_open:
                potential_json = llm_content[last_open:last_close + 1]
                logger.info(f"[PFU] Attempting to extract potential JSON from end of response")
                logger.debug(f"[PFU] Potential JSON: {potential_json[:200]}...")
                
                # Try to clean it up
                cleaned = potential_json.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    json_str = cleaned
                    used_pattern = "end-of-response extraction"
                    logger.info("[PFU] Extracted JSON from end of response")
            
            # Strategy 3: Look for any content that looks like JSON anywhere in the response
            if not json_str:
                logger.info("[PFU] Trying to find any JSON-like content in the response")
                # Look for any content that starts with { and ends with }
                json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_content, re.DOTALL)
                if json_matches:
                    # Take the longest match as it's most likely to be the complete plan
                    json_str = max(json_matches, key=len)
                    used_pattern = "anywhere-in-response extraction"
                    logger.info(f"[PFU] Found JSON-like content using {used_pattern}")
                    logger.debug(f"[PFU] Extracted content: {json_str[:200]}...")
        
        # Try to parse the JSON
        plan_data = None
        parsing_error = None
        
        if json_str:
            try:
                # Clean up common JSON issues
                cleaned_json = json_str
                
                # Remove any trailing commas before closing braces/brackets
                cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                
                # Fix unquoted property names - this is the main issue causing the current error
                cleaned_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                
                # Fix other common quote issues
                cleaned_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                
                logger.debug(f"[PFU] Attempting to parse cleaned JSON: {cleaned_json[:300]}...")
                plan_data = json.loads(cleaned_json)
                logger.info("[PFU] Successfully parsed JSON response")
                
            except json.JSONDecodeError as e:
                parsing_error = f"JSON parsing failed: {str(e)}"
                logger.error(f"[PFU] JSON parsing error: {e}")
                logger.debug(f"[PFU] Failed JSON string: {json_str}")
                
                # Try to fix common JSON issues more aggressively
                try:
                    # Remove any text before the first { and after the last }
                    first_brace = json_str.find('{')
                    last_brace = json_str.rfind('}')
                    if first_brace != -1 and last_brace != -1:
                        cleaned_json = json_str[first_brace:last_brace + 1]
                        
                        # More aggressive cleaning for unquoted property names
                        cleaned_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                        
                        # Remove trailing commas
                        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                        
                        # Try to fix any remaining quote issues
                        cleaned_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                        
                        # Additional cleaning for common LLM JSON issues
                        # Fix unquoted property names more comprehensively
                        cleaned_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                        
                        # Fix unquoted string values that should be quoted
                        cleaned_json = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_\s]*?)(?=\s*[,}])', r': "\1"', cleaned_json)
                        
                        # Fix any remaining unquoted property names (catch-all)
                        cleaned_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                        
                        # Remove any trailing commas before closing braces/brackets
                        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                        
                        # Fix any malformed quotes
                        cleaned_json = re.sub(r'([^"\\])\"([^"\\])', r'\1\\"\2', cleaned_json)
                        
                        logger.info(f"[PFU] Attempting to parse cleaned JSON: {cleaned_json[:200]}...")
                        
                        plan_data = json.loads(cleaned_json)
                        logger.info("[PFU] Successfully parsed JSON after aggressive cleaning")
                        parsing_error = None
                except json.JSONDecodeError as e:
                    parsing_error = f"JSON parsing failed even after cleaning: {str(e)}"
                    logger.warning(f"[PFU] JSON parsing failed after cleaning: {e}")
                    
                    # Try even more aggressive cleaning strategies
                    logger.info("[PFU] Attempting more aggressive JSON cleaning...")
                    
                    try:
                        # Strategy 1: Fix unquoted property names more aggressively
                        aggressive_cleaned = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', cleaned_json)
                        
                        # Strategy 2: Fix unquoted string values
                        aggressive_cleaned = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_\s]*?)(?=\s*[,}])', r': "\1"', aggressive_cleaned)
                        
                        # Strategy 3: Remove any problematic characters that might cause issues
                        aggressive_cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', aggressive_cleaned)
                        
                        # Strategy 4: Try to balance braces and brackets
                        open_braces = aggressive_cleaned.count('{')
                        close_braces = aggressive_cleaned.count('}')
                        open_brackets = aggressive_cleaned.count('[')
                        close_brackets = aggressive_cleaned.count(']')
                        
                        # Add missing closing braces/brackets if needed
                        if open_braces > close_braces:
                            aggressive_cleaned += '}' * (open_braces - close_braces)
                        if open_brackets > close_brackets:
                            aggressive_cleaned += ']' * (open_brackets - close_brackets)
                        
                        # Strategy 5: Fix the specific error pattern the user is seeing
                        # Look for patterns like: property: value (unquoted property names)
                        aggressive_cleaned = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^"\d{][^,}]*?)(?=\s*[,}])', r'\1 "\2": "\3"', aggressive_cleaned)
                        
                        # Strategy 6: Fix any remaining unquoted property names (final catch-all)
                        aggressive_cleaned = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', aggressive_cleaned)
                        
                        logger.info(f"[PFU] Attempting to parse aggressively cleaned JSON: {aggressive_cleaned[:200]}...")
                        
                        plan_data = json.loads(aggressive_cleaned)
                        logger.info("[PFU] Successfully parsed JSON after aggressive cleaning")
                        parsing_error = None
                        
                    except json.JSONDecodeError as e2:
                        parsing_error = f"JSON parsing failed even after aggressive cleaning: {str(e2)}"
                        logger.error(f"[PFU] JSON parsing failed even after aggressive cleaning: {e2}")
                        
                        # Final attempt: try to extract just the operations part
                        try:
                            logger.info("[PFU] Attempting to extract just operations from malformed JSON...")
                            
                            # Look for operations array specifically
                            operations_match = re.search(r'operations["\s]*:["\s]*\[(.*?)\]', aggressive_cleaned, re.DOTALL)
                            if operations_match:
                                operations_content = operations_match.group(1)
                                logger.info(f"[PFU] Found operations content: {operations_content[:100]}...")
                                
                                # Try to parse just the operations as a valid JSON array
                                operations_json = f'[{operations_content}]'
                                operations_array = json.loads(operations_json)
                                
                                # Create a minimal valid plan
                                plan_data = {
                                    "operations": operations_array,
                                    "objective": "Plan extracted from malformed response",
                                    "execution_strategy": "Operations were successfully extracted",
                                    "parsing_error": "Full plan parsing failed, but operations were extracted",
                                    "llm_response": llm_content[:500] + "..." if len(llm_content) > 500 else llm_content
                                }
                                
                                logger.info(f"[PFU] Successfully extracted {len(operations_array)} operations from malformed JSON")
                                parsing_error = None
                                
                        except Exception as e3:
                            parsing_error = f"Failed to extract operations: {str(e3)}"
                            logger.error(f"[PFU] Failed to extract operations: {e3}")
            except Exception as e:
                parsing_error = f"Unexpected parsing error: {str(e)}"
                logger.error(f"[PFU] Error parsing LLM response: {e}")
        
        # Validate the plan structure
        if plan_data:
            logger.info(f"[PFU] Plan data keys: {list(plan_data.keys())}")
            
            # Check for operations at root level (the restored format)
            if "operations" in plan_data:
                operations = plan_data["operations"]
                if isinstance(operations, list):
                    logger.info(f"[PFU] Found {len(operations)} operations at root level")
                    
                    # Extract operations from the complex step structure
                    extracted_operations = []
                    for i, step in enumerate(operations):
                        if isinstance(step, dict) and "operation" in step:
                            # Extract the actual operation from the step
                            operation_data = step["operation"]
                            if isinstance(operation_data, dict) and "op" in operation_data:
                                # This is the new complex format with step validation
                                extracted_operations.append(operation_data)
                                logger.debug(f"[PFU] Step {i+1}: {operation_data.get('op', 'unknown')} - {step.get('description', 'no description')}")
                            else:
                                # Fallback for simpler operation format
                                extracted_operations.append(operation_data)
                                logger.debug(f"[PFU] Step {i+1}: {operation_data.get('operation', 'unknown')} - {step.get('description', 'no description')}")
                        else:
                            logger.warning(f"[PFU] Step {i+1} is invalid: {step}")
                    
                    # Update operations with extracted ones
                    plan_data["operations"] = extracted_operations
                    logger.info(f"[PFU] Extracted {len(extracted_operations)} operations from complex plan structure")
                    
                else:
                    logger.warning(f"[PFU] Operations at root level is not a list: {type(operations)}")
                    plan_data["operations"] = []
            
            # Fallback: Handle nested "plan" structure if it exists (for backward compatibility)
            elif "plan" in plan_data and isinstance(plan_data["plan"], dict):
                logger.info("[PFU] Found nested 'plan' structure, extracting operations from it")
                nested_plan = plan_data["plan"]
                if "operations" in nested_plan:
                    operations = nested_plan["operations"]
                    if isinstance(operations, list):
                        logger.info(f"[PFU] Found {len(operations)} operations in nested plan")
                        
                        # Extract operations from the complex step structure
                        extracted_operations = []
                        for i, step in enumerate(operations):
                            if isinstance(step, dict) and "operation" in step:
                                # Extract the actual operation from the step
                                operation_data = step["operation"]
                                if isinstance(operation_data, dict) and "op" in operation_data:
                                    # This is the new complex format with step validation
                                    extracted_operations.append(operation_data)
                                    logger.debug(f"[PFU] Step {i+1}: {operation_data.get('op', 'unknown')} - {step.get('description', 'no description')}")
                                else:
                                    # Fallback for simpler operation format
                                    extracted_operations.append(operation_data)
                                    logger.debug(f"[PFU] Step {i+1}: {operation_data.get('operation', 'unknown')} - {step.get('description', 'no description')}")
                            else:
                                logger.warning(f"[PFU] Step {i+1} is invalid: {step}")
                        
                        # Update operations with extracted ones
                        plan_data["operations"] = extracted_operations
                        logger.info(f"[PFU] Extracted {len(extracted_operations)} operations from nested plan")
                        
                        # Also copy other useful fields from nested plan
                        if "objective" in nested_plan:
                            plan_data["objective"] = nested_plan["objective"]
                        if "execution_strategy" in nested_plan:
                            plan_data["execution_strategy"] = nested_plan["execution_strategy"]
                        if "current_state_analysis" in nested_plan:
                            plan_data["current_state_analysis"] = nested_plan["current_state_analysis"]
                        if "required_changes" in nested_plan:
                            plan_data["required_changes"] = nested_plan["required_changes"]
                        if "validation_strategy" in nested_plan:
                            plan_data["validation_strategy"] = nested_plan["validation_strategy"]
                        
                    else:
                        logger.warning(f"[PFU] Operations in nested plan is not a list: {type(operations)}")
                        plan_data["operations"] = []
                else:
                    logger.warning("[PFU] Nested plan missing 'operations' key")
                    plan_data["operations"] = []
            else:
                logger.warning("[PFU] Plan missing 'operations' key at both root and nested levels")
                plan_data["operations"] = []
        
        # If parsing failed or no valid plan, create a fallback
        if not plan_data or not plan_data.get("operations"):
            logger.error("[PFU] No valid plan data could be parsed from LLM response")
            
            # Extract any useful information from the LLM response
            objective = "Plan parsing failed - see LLM response for details"
            strategy = "Unable to parse structured plan"
            
            # Try to extract objective and strategy from the text
            if llm_content:
                # Look for common patterns
                obj_match = re.search(r'objective["\s]*:["\s]*([^"\n]+)', llm_content, re.IGNORECASE)
                if obj_match:
                    objective = obj_match.group(1).strip()
                
                strat_match = re.search(r'strategy["\s]*:["\s]*([^"\n]+)', llm_content, re.IGNORECASE)
                if strat_match:
                    strategy = strat_match.group(1).strip()
            
            plan_data = {
                "objective": objective,
                "execution_strategy": strategy,
                "operations": [],
                "parsing_error": parsing_error or "Failed to parse structured plan from LLM response",
                "llm_response": llm_content[:1000] + "..." if len(llm_content) > 1000 else llm_content
            }
        
        logger.info(f"[PFU] Final plan_data structure: {plan_data}")
        logger.info(f"[PFU] Final plan_data keys: {list(plan_data.keys())}")
        logger.info(f"[PFU] Final operations: {plan_data.get('operations', 'none')}")
        
        return plan_data

    def execute_plan_step_by_step(self, plan: dict, flow_data: dict, max_steps: int = 10) -> dict:
        """Execute a PFU plan step by step with validation between steps.
        
        This method processes operations incrementally, similar to how Cursor handles code changes.
        Each step is validated before proceeding to the next one.
        """
        if not plan or "operations" not in plan:
            raise ValueError("Invalid plan: missing operations")
        
        operations = plan["operations"]
        if not operations:
            logger.info("[PFU] No operations to execute")
            return flow_data
        
        if len(operations) > max_steps:
            logger.warning(f"[PFU] Plan has {len(operations)} operations, limiting to {max_steps}")
            operations = operations[:max_steps]
        
        current_data = flow_data.copy()
        executed_steps = []
        failed_steps = []
        
        logger.info(f"[PFU] Executing plan with {len(operations)} operations")
        
        for i, step in enumerate(operations):
            try:
                logger.info(f"[PFU] Executing step {i+1}/{len(operations)}: {step.get('description', 'Unknown')}")
                
                # Extract the operation from the step
                operation = step.get("operation")
                if not operation:
                    logger.warning(f"[PFU] Step {i+1} missing operation, skipping")
                    continue
                
                # Apply the operation
                from langflow.api.v1.flows import _apply_flow_operations
                new_data = _apply_flow_operations(current_data, [operation])
                
                # Validate the step if validation criteria are provided
                validation = step.get("validation")
                if validation:
                    logger.info(f"[PFU] Validating step {i+1}: {validation}")
                    # Here you could add custom validation logic
                    # For now, we'll just check if the operation was applied
                    if new_data == current_data:
                        raise ValueError(f"Operation {operation.get('op')} did not modify the flow")
                
                # Update current data
                current_data = new_data
                executed_steps.append({
                    "step": i + 1,
                    "operation": operation,
                    "description": step.get("description"),
                    "status": "success"
                })
                
                logger.info(f"[PFU] Step {i+1} completed successfully")
                
            except Exception as e:
                logger.error(f"[PFU] Step {i+1} failed: {e}")
                failed_steps.append({
                    "step": i + 1,
                    "operation": step.get("operation"),
                    "description": step.get("description"),
                    "status": "failed",
                    "error": str(e)
                })
                
                # For now, we'll continue with other steps
                # In a production system, you might want to rollback or stop here
                continue
        
        # Log execution summary
        success_count = len(executed_steps)
        failure_count = len(failed_steps)
        logger.info(f"[PFU] Execution completed: {success_count} successful, {failure_count} failed")
        
        if failed_steps:
            logger.warning(f"[PFU] Failed steps: {[s['step'] for s in failed_steps]}")
        
        return {
            "final_flow_data": current_data,
            "execution_summary": {
                "total_steps": len(operations),
                "successful_steps": success_count,
                "failed_steps": failure_count,
                "executed_steps": executed_steps,
                "failed_steps": failed_steps
            }
        }

    def build_docs_from_types(self, all_types_dict: dict) -> list[dict[str, Any]]:
        """Transform components types dict into comprehensive docs for indexing."""
        docs: list[dict[str, Any]] = []
        components = all_types_dict if isinstance(all_types_dict, dict) else {}
        # expected structure: {"components": {type: {name: {template...}}}}
        comps = components.get("components", {})
        
        for comp_type, entries in comps.items():
            for name, meta in entries.items():
                tmpl = meta.get("template", {})
                base_cls = meta.get("base_classes") or meta.get("node", {}).get("base_classes")
                display_name = meta.get("display_name", name)
                description = meta.get("description", "")
                
                # Extract key template information
                template_keys = list(tmpl.keys())
                input_fields = []
                output_fields = []
                
                # Look for input/output information
                if "outputs" in meta:
                    output_fields = [out.get("name", "") for out in meta.get("outputs", [])]
                if "inputs" in meta:
                    input_fields = [inp.get("name", "") for inp in meta.get("inputs", [])]
                
                # Create comprehensive component documentation
                text = f"""
Component: {name}
Display Name: {display_name}
Type: {comp_type}
Description: {description}
Base Classes: {base_cls}
Template Fields: {template_keys}
Input Fields: {input_fields}
Output Fields: {output_fields}
Template Structure: {json.dumps(tmpl, indent=2)}
"""
                
                docs.append({
                    "id": f"{comp_type}:{name}", 
                    "text": text, 
                    "meta": {
                        "type": comp_type, 
                        "name": name,
                        "display_name": display_name,
                        "description": description,
                        "base_classes": base_cls,
                        "template_keys": template_keys,
                        "input_fields": input_fields,
                        "output_fields": output_fields
                    }
                })
        
        return docs

    def index_components(self, all_types_dict: dict, reset: bool = False) -> int:
        """Index components into the vector database.
        
        Args:
            all_types_dict: Dictionary containing component definitions
            reset: If True, clear the database before indexing. Default False for persistence.
        """
        docs = self.build_docs_from_types(all_types_dict)
        if reset:
            self.retriever.reset()
        return self.retriever.upsert(docs)

    def retrieve(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        return self.retriever.query(query, k=k)

    # ---------- File indexing ----------
    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
        if chunk_size <= 0:
            return [text]
        step = max(1, chunk_size - max(0, overlap))
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    def index_files(
        self,
        file_patterns: list[str],
        *,
        reset: bool = False,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> dict[str, Any]:
        """Index files into the local vector DB.

        file_patterns: list of file paths or glob patterns.
        reset: if True, clears the collection before upserting.
        chunk_size/overlap: controls text chunking for better recall.
        """
        if reset:
            self.retriever.reset()

        # Expand globs and deduplicate paths
        paths: set[str] = set()
        for pattern in file_patterns:
            matched = glob.glob(pattern, recursive=True)
            if not matched and os.path.isfile(pattern):
                matched = [pattern]
            for p in matched:
                if os.path.isfile(p):
                    paths.add(os.path.abspath(p))

        total_files = 0
        total_chunks = 0
        batch: list[dict[str, Any]] = []

        for fpath in sorted(paths):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[PFU-RAG] Skipping file %s due to read error: %s", fpath, exc)
                continue
            total_files += 1
            chunks = self._chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{fpath}:{idx}"
                batch.append({"id": doc_id, "text": chunk, "meta": {"path": fpath, "chunk": idx}})
            total_chunks += len(chunks)

        if batch:
            self.retriever.upsert(batch)

        return {"files": total_files, "chunks": total_chunks}


