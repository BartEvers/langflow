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
        
        # Compose the final prompt
        final_prompt = "\n".join(parts)
        
        # CLI logging - show the complete prompt that goes to LLM
        print("\n" + "="*100)
        print("🚀 COMPLETE PROMPT SENT TO LLM")
        print("="*100)
        print(final_prompt)
        print("="*100)
        print(f"📊 PROMPT STATISTICS:")
        print(f"   Total length: {len(final_prompt)} characters")
        print(f"   Total lines: {final_prompt.count(chr(10)) + 1}")
        print(f"   Flow data included: {'Yes' if flow_data else 'No'}")
        print(f"   Snippets included: {len(retrieved_snippets) if retrieved_snippets else 0}")
        print("="*100 + "\n")
        
        return final_prompt

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

    # ---------- Step-by-Step PFU Execution ----------
    
    # ---------- Dynamic Step-by-Step PFU Execution ----------
    
    def execute_step_by_step_pfu(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> dict[str, Any]:
        """Execute PFU with direct response using <NAME> format for component addition."""
        logger.info(f"[PFU] Starting direct PFU execution for: {user_prompt[:100]}...")
        
        # Generate direct response with component tags if needed
        direct_response = self._generate_direct_response(user_prompt, flow_data, retrieved_snippets)
        
        return {
            "status": "completed",
            "message": direct_response,
            "total_steps": 1,
            "completed_steps": 1,
            "execution_summary": {
                "total_steps": 1,
                "completed_steps": 1,
                "status": "completed"
            }
        }
    
    def _generate_direct_response(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> str:
        """Generate a direct response with component tags if needed."""
        # Compose a simple, direct prompt
        direct_prompt = self._compose_direct_prompt(user_prompt, flow_data, retrieved_snippets)
        
        try:
            llm_response = self._call_llm(direct_prompt)
            return llm_response
        except Exception as e:
            logger.error(f"[PFU] Direct response generation failed: {e}")
            return f"I understand you want to {user_prompt.lower()}. Please try rephrasing your request to be more specific about what component you'd like me to add."
    
    def _compose_direct_prompt(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> str:
        """Compose a simple, direct prompt for immediate response."""
        parts = [
            "You are Airelius, a helpful Langflow assistant. Respond directly to the user's request.",
            "",
            "IMPORTANT RULES:",
            "1. If the user wants to add a component, respond with <COMPONENT_NAME>",
            "2. Use the exact component name from Langflow (e.g., <ChatInput>, <OpenAI>, <TextLoader>)",
            "3. Keep your response simple and direct",
            "4. If you need to add multiple components, use multiple tags: <Component1> <Component2>",
            "5. If the request is unclear, ask for clarification",
            "",
            "USER REQUEST:",
            user_prompt,
            "",
            "AVAILABLE COMPONENTS:",
        ]
        
        if retrieved_snippets:
            parts.append(f"Reference components: {len(retrieved_snippets)} available")
            for snippet in retrieved_snippets[:3]:  # Show first 3 for reference
                text = snippet.get('text', '')[:100]
                parts.append(f"- {text}...")
        
        if flow_data:
            nodes = flow_data.get("nodes", [])
            edges = flow_data.get("edges", [])
            parts.append(f"\nCURRENT FLOW: {len(nodes)} nodes, {len(edges)} edges")
        
        parts.extend([
            "",
            "RESPOND NOW with either:",
            "1. <ComponentName> if adding a component",
            "2. A helpful explanation if no component needed",
            "3. A question if the request is unclear"
        ])
        
        final_prompt = "\n".join(parts)
        
        # CLI logging for complexity analysis prompt
        print(f"\n{'='*80}")
        print(f"🧠 COMPLEXITY ANALYSIS PROMPT")
        print(f"{'='*80}")
        print(final_prompt)
        print(f"{'='*80}")
        print(f"📊 COMPLEXITY PROMPT STATISTICS:")
        print(f"   Total length: {len(final_prompt)} characters")
        print(f"   Total lines: {final_prompt.count(chr(10)) + 1}")
        print(f"   Flow data included: {'Yes' if flow_data else 'No'}")
        print(f"   Snippets included: {len(retrieved_snippets) if retrieved_snippets else 0}")
        print(f"{'='*80}\n")
        
        return final_prompt
    
    def _analyze_request_complexity(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze the user's request and determine what planning steps are needed."""
        complexity_prompt = self._compose_complexity_analysis_prompt(user_prompt, flow_data, retrieved_snippets)
        
        try:
            print(f"⏳ Calling LLM for complexity analysis...")
            llm_response = self._call_llm(complexity_prompt)
            
            print(f"\n{'='*80}")
            print(f"🧠 COMPLEXITY ANALYSIS LLM RESPONSE")
            print(f"{'='*80}")
            print(llm_response)
            print(f"{'='*80}")
            print(f"📊 COMPLEXITY RESPONSE STATISTICS:")
            print(f"   Total length: {len(llm_response)} characters")
            print(f"   Total lines: {llm_response.count(chr(10)) + 1}")
            print(f"   Contains JSON: {'Yes' if '{' in llm_response and '}' in llm_response else 'No'}")
            print(f"{'='*80}\n")
            
            complexity_data = self._parse_complexity_response(llm_response)
            
            # Ensure we have a valid step plan
            if not complexity_data.get("required_steps"):
                logger.warning("[PFU] LLM didn't provide required_steps, using fallback plan")
                complexity_data["required_steps"] = self._get_fallback_step_plan(user_prompt)
            
            return complexity_data
            
        except Exception as e:
            logger.error(f"[PFU] Complexity analysis failed: {e}, using fallback plan")
            return self._get_fallback_step_plan(user_prompt)
    
    def _compose_complexity_analysis_prompt(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> str:
        """Compose a prompt for analyzing request complexity and determining required steps."""
        parts = [
            "You are Airelius, an expert Langflow flow architect. Your task is to analyze a user's request and determine what planning steps are needed.",
            "",
            "ANALYZE THIS REQUEST:",
            f"User Request: {user_prompt}",
            "",
            "AVAILABLE CONTEXT:",
        ]
        
        if flow_data:
            try:
                nodes = flow_data.get("nodes", [])
                edges = flow_data.get("edges", [])
                flow_summary = f"Flow has {len(nodes)} nodes and {len(edges)} edges"
                if nodes:
                    node_types = [node.get("type", "unknown") for node in nodes[:3]]
                    flow_summary += f". Node types: {', '.join(node_types)}"
                    if len(nodes) > 3:
                        flow_summary += f" and {len(nodes) - 3} more"
                parts.append(f"Current Flow: {flow_summary}")
            except Exception as e:
                parts.append(f"Current Flow: Available but complex (error parsing: {e})")
        else:
            parts.append("Current Flow: None (new flow)")
        
        if retrieved_snippets:
            parts.append(f"Available Components: {len(retrieved_snippets)} component definitions loaded")
        
        parts.extend([
            "",
            "STEP PLANNING RULES:",
            "1. Simple requests (1-2 steps): Direct execution, minimal planning",
            "2. Medium requests (3-4 steps): Basic analysis + execution",
            "3. Complex requests (5+ steps): Full strategic planning + execution",
            "",
            "AVAILABLE STEP TYPES:",
            "- analyze_flow: Analyze current flow structure and capabilities",
            "- plan_strategy: Create strategic approach for the request",
            "- design_solution: Design the specific solution",
            "- validate_approach: Validate the proposed approach",
            "- create_operations: Generate executable operations",
            "- quality_check: Final quality and validation check",
            "- direct_execution: Execute simple changes directly",
            "",
            "RESPONSE FORMAT (MUST BE VALID JSON):",
            "{",
            '  "complexity_level": "simple|medium|complex",',
            '  "reasoning": "Why this complexity level was chosen",',
            '  "required_steps": [',
            '    {',
            '      "name": "step_name",',
            '      "description": "What this step will do",',
            '      "type": "step_type",',
            '      "priority": "high|medium|low"',
            '    }',
            '  ]',
            '}',
            "",
            "Generate the complexity analysis now:"
        ])
        
        return "\n".join(parts)
    
    def _parse_complexity_response(self, response: str) -> dict[str, Any]:
        """Parse the complexity analysis response from the LLM."""
        try:
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
                
                parsed = json.loads(json_str)
                return parsed
            else:
                logger.warning("[PFU] No JSON found in complexity response, using fallback")
                return self._get_fallback_step_plan("")
                
        except Exception as e:
            logger.warning(f"[PFU] Failed to parse complexity response: {e}, using fallback")
            return self._get_fallback_step_plan("")
    
    def _get_fallback_step_plan(self, user_prompt: str) -> dict[str, Any]:
        """Provide a fallback step plan if the LLM analysis fails."""
        # Simple fallback based on prompt keywords
        prompt_lower = user_prompt.lower()
        
        if any(word in prompt_lower for word in ["add", "create", "insert", "new"]):
            # Simple addition - 2 steps
            return {
                "complexity_level": "simple",
                "reasoning": "Fallback: Simple addition request detected",
                "required_steps": [
                    {
                        "name": "analyze_flow",
                        "description": "Analyze current flow structure",
                        "type": "analyze_flow",
                        "priority": "high"
                    },
                    {
                        "name": "direct_execution", 
                        "description": "Execute the addition directly",
                        "type": "direct_execution",
                        "priority": "high"
                    }
                ]
            }
        elif any(word in prompt_lower for word in ["refactor", "restructure", "redesign", "complex"]):
            # Complex refactoring - 5 steps
            return {
                "complexity_level": "complex",
                "reasoning": "Fallback: Complex refactoring request detected",
                "required_steps": [
                    {
                        "name": "analyze_flow",
                        "description": "Analyze current flow structure",
                        "type": "analyze_flow",
                        "priority": "high"
                    },
                    {
                        "name": "plan_strategy",
                        "description": "Create strategic refactoring plan",
                        "type": "plan_strategy",
                        "priority": "high"
                    },
                    {
                        "name": "design_solution",
                        "description": "Design the refactored solution",
                        "type": "design_solution",
                        "priority": "high"
                    },
                    {
                        "name": "create_operations",
                        "description": "Generate executable operations",
                        "type": "create_operations",
                        "priority": "high"
                    },
                    {
                        "name": "quality_check",
                        "description": "Validate the refactored solution",
                        "type": "quality_check",
                        "priority": "medium"
                    }
                ]
            }
        else:
            # Default medium complexity - 3 steps
            return {
                "complexity_level": "medium",
                "reasoning": "Fallback: Default medium complexity plan",
                "required_steps": [
                    {
                        "name": "analyze_flow",
                        "description": "Analyze current flow structure",
                        "type": "analyze_flow",
                        "priority": "high"
                    },
                    {
                        "name": "design_solution",
                        "description": "Design the solution",
                        "type": "design_solution",
                        "priority": "high"
                    },
                    {
                        "name": "create_operations",
                        "description": "Generate executable operations",
                        "type": "create_operations",
                        "priority": "high"
                    }
                ]
            }
    
    def _execute_dynamic_step(self, step_name: str, step_description: str, user_prompt: str, 
                             flow_data: dict | None, retrieved_snippets: list[dict[str, Any]], 
                             previous_steps: dict[str, Any], step_config: dict[str, Any]) -> dict[str, Any]:
        """Execute a single dynamic step based on its type."""
        step_type = step_config.get("type", "unknown")
        
        logger.info(f"[PFU] Executing dynamic step: {step_name} (type: {step_type})")
        
        # Compose step-specific prompt
        step_prompt = self._compose_dynamic_step_prompt(
            step_name, step_description, step_type, user_prompt, 
            flow_data, retrieved_snippets, previous_steps
        )
        
        # Execute the step
        step_response = self._call_llm(step_prompt)
        step_data = self._parse_step_response(step_response, step_name)
        
        # Add metadata
        step_data["step_type"] = step_type
        step_data["step_description"] = step_description
        step_data["step_config"] = step_config
        
        return step_data
    
    def _compose_dynamic_step_prompt(self, step_name: str, step_description: str, step_type: str,
                                   user_prompt: str, flow_data: dict | None, 
                                   retrieved_snippets: list[dict[str, Any]], 
                                   previous_steps: dict[str, Any]) -> str:
        """Compose a prompt for a specific dynamic step."""
        parts = [
            f"You are Airelius, executing step: {step_name}",
            f"Step Description: {step_description}",
            f"Step Type: {step_type}",
            "",
            f"USER REQUEST: {user_prompt}",
            ""
        ]
        
        # Add flow context
        if flow_data:
            try:
                nodes = flow_data.get("nodes", [])
                edges = flow_data.get("edges", [])
                flow_summary = f"Flow has {len(nodes)} nodes and {len(edges)} edges"
                if nodes:
                    node_types = [node.get("type", "unknown") for node in nodes[:3]]
                    flow_summary += f". Node types: {', '.join(node_types)}"
                    if len(nodes) > 3:
                        flow_summary += f" and {len(nodes) - 3} more"
                parts.append(f"CURRENT FLOW: {flow_summary}")
            except Exception as e:
                parts.append(f"CURRENT FLOW: Available but complex (error parsing: {e})")
        else:
            parts.append("CURRENT FLOW: None (new flow)")
        
        # Add component snippets
        if retrieved_snippets:
            parts.append("\nRELEVANT COMPONENTS:")
            for snippet in retrieved_snippets[:5]:
                parts.append(f"- {snippet.get('text', '')[:200]}...")
        
        # Add previous steps context
        if previous_steps:
            parts.append("\nPREVIOUS STEPS CONTEXT:")
            for prev_step_name, prev_step_data in previous_steps.items():
                parts.append(f"- {prev_step_name}: {str(prev_step_data.get('response', ''))[:100]}...")
        
        # Add step-specific instructions
        if step_type == "analyze_flow":
            parts.extend([
                "",
                "ANALYZE THE CURRENT FLOW:",
                "1. Understand the existing structure",
                "2. Identify current capabilities",
                "3. Note any limitations or issues",
                "4. Provide a clear analysis summary"
            ])
        elif step_type == "plan_strategy":
            parts.extend([
                "",
                "CREATE A STRATEGIC PLAN:",
                "1. Define the overall approach",
                "2. Identify key considerations",
                "3. Plan the execution sequence",
                "4. Consider potential challenges"
            ])
        elif step_type == "design_solution":
            parts.extend([
                "",
                "DESIGN THE SOLUTION:",
                "1. Create the specific solution design",
                "2. Consider component requirements",
                "3. Plan the implementation details",
                "4. Ensure it meets the user's needs"
            ])
        elif step_type == "create_operations":
            parts.extend([
                "",
                "GENERATE EXECUTABLE OPERATIONS:",
                "1. Create valid Langflow operations",
                "2. Use proper operation format",
                "3. Include all necessary details",
                "4. Ensure operations are executable"
            ])
        elif step_type == "quality_check":
            parts.extend([
                "",
                "PERFORM QUALITY CHECK:",
                "1. Validate the proposed solution",
                "2. Check for potential issues",
                "3. Ensure quality standards",
                "4. Provide validation summary"
            ])
        elif step_type == "direct_execution":
            parts.extend([
                "",
                "EXECUTE DIRECTLY:",
                "1. Implement the requested change",
                "2. Use the simplest approach", 
                "3. Ensure it works correctly",
                "4. Provide execution summary",
                "5. If adding a component, use: <NAME>"
            ])
        
        parts.extend([
            "",
            "RESPONSE FORMAT: Provide a clear, helpful response that accomplishes this step. Make it around 50 words. Don't make bullet points, bullet lists, or numbered lists. Just write out the steps short and concise.",
            "If this step should add a component, use this format: <NAME>. It is REALLY important that you add <NAME> if a component is needed. DO NOT FORGET OR SKIP THIS STEP. Also keep to the format <NAME> and do not add any other text or comments, especially not adding something like COMPONENT, COMPONENT_NAME, COMPONENT_ID, etc.",
            "If this step should add a component, name the ID of the component(s) its supposed to connect to, use this format: <CONNECT_TO_ID>",
            "Use the official name like, PromptLayerOpenAI, ConversationSummaryBufferMemory, TextLoader, Chroma, OpenAIEmbeddings, GuardrailsComponent",
            "Example: <PromptLayerOpenAI> or <ConversationSummaryBufferMemory>",
            "",
            "Execute this step now:"
        ])
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a specific prompt."""
        try:
            from openai import OpenAI
            import os
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[PFU] LLM call failed: {e}")
            return f"Error calling LLM: {str(e)}"

    async def chat(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> str:
        """Generate a conversational response for chat mode."""
        try:
            # Compose a chat-focused prompt
            chat_prompt = self._compose_chat_prompt(user_prompt, flow_data, retrieved_snippets)
            
            # Call LLM for chat response
            chat_response = await self._call_llm_async(chat_prompt)
            return chat_response
        except Exception as e:
            logger.error(f"[PFU] Chat failed: {e}")
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

    def _compose_chat_prompt(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]]) -> str:
        """Compose a prompt specifically for chat mode."""
        parts = []
        
        # Add chat context
        parts.append("You are an AI assistant that helps users understand and work with Langflow flows. You can:")
        parts.append("- Explain what flows do")
        parts.append("- Suggest improvements")
        parts.append("- Help debug issues")
        parts.append("- Provide guidance on best practices")
        parts.append("- Answer questions about Langflow components")
        parts.append("\nPlease provide helpful, conversational responses. Be concise but informative.")
        
        # Add RAG context for component understanding
        if retrieved_snippets:
            parts.append("\n" + "="*60)
            parts.append("RELEVANT COMPONENTS FOR THIS REQUEST:")
            parts.append("="*60)
            
            for i, snippet in enumerate(retrieved_snippets[:5]):  # Limit to 5 for chat
                text = snippet.get('text', '')
                metadata = snippet.get('metadata', {})
                
                component_info = f"\n--- Component {i+1} ---\n"
                
                if metadata:
                    if 'component_type' in metadata:
                        component_info += f"Type: {metadata['component_type']}\n"
                    if 'display_name' in metadata:
                        component_info += f"Name: {metadata['display_name']}\n"
                    if 'description' in metadata:
                        component_info += f"Description: {metadata['description']}\n"
                
                if len(text) > 200:  # Shorter for chat
                    component_info += f"Code: {text[:200]}...\n"
                else:
                    component_info += f"Code: {text}\n"
                
                parts.append(component_info)
        
        # Add current flow context if available
        if flow_data:
            try:
                nodes = flow_data.get("nodes", [])
                edges = flow_data.get("edges", [])
                flow_summary = f"Current flow has {len(nodes)} nodes and {len(edges)} edges"
                if nodes:
                    node_types = [node.get("type", "unknown") for node in nodes[:3]]  # Show first 3 node types
                    flow_summary += f". Node types: {', '.join(node_types)}"
                    if len(nodes) > 3:
                        flow_summary += f" and {len(nodes) - 3} more"
                
                parts.append(f"\nCURRENT FLOW: {flow_summary}")
            except Exception as e:
                logger.warning(f"[PFU] Failed to process flow data for chat: {e}")
        
        # Add the user's question
        parts.append(f"\nUSER QUESTION: {user_prompt}")
        parts.append("\nPlease provide a helpful response based on the context above.")
        
        return "\n".join(parts)
    
    def _parse_step_response(self, response: str, step_name: str) -> dict[str, Any]:
        """Parse the response from a specific step."""
        try:
            # Try to extract JSON from the response
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
                
                parsed = json.loads(json_str)
                # Ensure we always have a response field
                if "response" not in parsed:
                    parsed["response"] = response
                return parsed
            else:
                # If no JSON found, return the response as text
                return {"response": response, "step": step_name}
        except Exception as e:
            logger.warning(f"[PFU] Failed to parse step {step_name} response: {e}")
            return {"response": response, "step": step_name, "parse_error": str(e)}

    # ---------- Async Step-by-Step PFU Execution ----------
    
    async def execute_single_step(self, step_name: str, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]], previous_steps: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a single PFU step asynchronously using the dynamic step system."""
        logger.info(f"[PFU] Executing async step: {step_name}")
        
        print(f"\n{'='*80}")
        print(f"⚡ EXECUTING STEP: {step_name.upper()}")
        print(f"{'='*80}")
        print(f"📝 User Prompt: {user_prompt}")
        print(f"🔄 Flow Data Available: {'Yes' if flow_data else 'No'}")
        print(f"📊 Previous Steps: {len(previous_steps) if previous_steps else 0}")
        print(f"📋 Snippets Available: {len(retrieved_snippets) if retrieved_snippets else 0}")
        print(f"{'='*80}\n")
        
        # For backward compatibility, we'll create a step config based on the step name
        step_config = {
            "name": step_name,
            "description": f"Execute {step_name} step",
            "type": self._map_legacy_step_name(step_name),
            "priority": "high"
        }
        
        # Use the dynamic step execution system
        step_result = self._execute_dynamic_step(
            step_name, step_config["description"], user_prompt, flow_data,
            retrieved_snippets, previous_steps or {}, step_config
        )
        
        print(f"\n{'='*80}")
        print(f"✅ STEP COMPLETED: {step_name.upper()}")
        print(f"{'='*80}")
        print(f"📊 Result Type: {type(step_result)}")
        if isinstance(step_result, dict):
            print(f"📋 Result Keys: {list(step_result.keys())}")
            if 'response' in step_result:
                response_preview = str(step_result['response'])[:200]
                print(f"📝 Response Preview: {response_preview}...")
        print(f"{'='*80}\n")
        
        return step_result
    
    def _map_legacy_step_name(self, step_name: str) -> str:
        """Map legacy step names to new dynamic step types for backward compatibility."""
        legacy_mapping = {
            "strategic_planning": "plan_strategy",
            "objective": "design_solution", 
            "current_state_analysis": "analyze_flow",
            "required_changes": "design_solution",
            "execution_strategy": "plan_strategy",
            "step_design": "design_solution",
            "component_quality_validation": "quality_check"
        }
        return legacy_mapping.get(step_name, "design_solution")
    
    async def _call_llm_async(self, prompt: str) -> str:
        """Call the LLM with a specific prompt asynchronously."""
        try:
            from openai import AsyncOpenAI
            import os
            
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[PFU] Async LLM call failed: {e}")
            return f"Error calling LLM: {str(e)}"