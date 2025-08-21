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
    
    def execute_step_by_step_pfu(self, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]], available_templates: list[str] | None = None) -> dict[str, Any]:
        """Execute PFU with direct response using <NAME:RANDOM_ID> or <NAME:RANDOM_ID:EXISTING_ID1,EXISTING_ID2> format for component addition."""
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
            "1. If the user wants to add a component, respond with <COMPONENT_NAME-GENERATED_ID>",
            "2. Use the exact component name from Langflow (e.g., <ChatInput-abc123>, <OpenAI-def456>, <TextLoader-ghi789>)",
            "3. Keep your response simple and direct",
            "4. If you need to add multiple components, use multiple tags: <Component1-id1> <Component2-id2>",
            "5. For components that need connections, use: <COMPONENT_NAME-GENERATED_ID:EXISTING_COMPONENT_ID>",
            "6. For multiple connections, separate with commas: <COMPONENT_NAME-GENERATED_ID:EXISTING_ID1,EXISTING_ID2>",
            "7. If the request is unclear, ask for clarification",
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
            parts.append(f"FLOW DATA: {json.dumps(flow_data, indent=2)}")
        
        parts.extend([
            "",
            "RESPOND NOW with either:",
            "1. <ComponentName-GeneratedId> if adding a component without connections",
            "2. <ComponentName-GeneratedId:ExistingComponentId> if adding a component with one connection",
            "3. <ComponentName-GeneratedId:ExistingId1,ExistingId2> if adding a component with multiple connections",
            "4. A helpful explanation if no component needed",
            "5. A question if the request is unclear",
            "",
            "Examples:",
            "- <ChatInput-abc123> (no connections)",
            "- <WebSearchNoAPI-xyz789:ChatInput-GxsOd> (single connection)",
            "- <MemoryStore-ghi789:ChatInput-abc123,PromptTemplate-def456> (multiple connections)"
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
                             previous_steps: dict[str, Any], step_config: dict[str, Any], 
                             available_templates: list[str] | None = None) -> dict[str, Any]:
        """Execute a single dynamic step based on its type."""
        step_type = step_config.get("type", "unknown")
        
        logger.info(f"[PFU] Executing dynamic step: {step_name} (type: {step_type})")
        
        # Compose step-specific prompt
        step_prompt = self._compose_dynamic_step_prompt(
            step_name, step_description, step_type, user_prompt, 
            flow_data, retrieved_snippets, previous_steps, available_templates
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
                                   previous_steps: dict[str, Any],
                                   available_templates: list[str] | None = None) -> str:
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
                parts.append(f"FLOW DATA: {json.dumps(flow_data, indent=2)}")
            except Exception as e:
                parts.append(f"CURRENT FLOW: Available but complex (error parsing: {e})")
        else:
            parts.append("CURRENT FLOW: None (new flow)")
        
        # Add component snippets
        if retrieved_snippets:
            parts.append("\nRELEVANT COMPONENTS:")
            for snippet in retrieved_snippets[:5]:
                parts.append(f"- {snippet.get('text', '')[:200]}...")
        
        # Add available templates list
        if available_templates and len(available_templates) > 0:
            parts.append("\nAVAILABLE TEMPLATES:")
            for template in available_templates:
                parts.append(f"- {template}")
        else:
            parts.append("\nAVAILABLE TEMPLATES: None available")
        
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
                "5. If adding a component, use: <NAME:RANDOM_COMPONENT_ID> or <NAME:RANDOM_COMPONENT_ID:EXISTING_ID1,EXISTING_ID2> for multiple connections"
            ])
        
        parts.extend([
            "",
            "RESPONSE FORMAT: Provide a clear, helpful response that accomplishes this step. Make it around 50 words. Dont make bullet points, bullet lists, or numbered lists. Just write out the steps short and concise.",
            "If this step should add a component, use this format: <NAME:RANDOM_COMPONENT_ID: EDGE1[, EDGE2, ...]>.",
            "NAME is the component type (e.g., WebSearchNoAPI, ChatInput, PromptLayerOpenAI).",
            "Only use a component type that exists in the available templates; if not present, pick the closest available type.",
            "RANDOM_COMPONENT_ID is a new ID you generate (e.g., WebSearchNoAPI-xn3k3x).",
            "Inspect the flow to find the best existing component(s) to connect to.",
            "EXISTING ids must be real node IDs from the flow. Never modify or annotate IDs.",
            "EDGE SYNTAX (React Flow handle-sides): write every edge as 'SRC_NODE.out:<handle> -> DST_NODE.in:<field>'.",
            "Right-side handles are always sources (outputs). Left-side handles are always targets (inputs). React Flow enforces this regardless of drag direction.",
            "Do NOT use block-level annotations like '@out=...' and do NOT use 'dir=' anywhere.",
            "Resolve REAL handle names and types from the registry for BOTH ends before emitting an edge.",
            "TYPE SAFETY: ensure source output_types are compatible with target inputTypes. If not, insert the minimal adapter component and wire through it (emit an additional <ADAPTER:ID: ...> block).",
            "INSERT BETWEEN PROMPT AND LLM: emit exactly two edges in this order: 'Prompt-<PROMPT_ID>.out:prompt -> <NEW_ID>.in:<input_from_registry>', then '<NEW_ID>.out:<output_from_registry> -> LanguageModelComponent-<LLM_ID>.in:system_message'.",
            "ID & HANDLE CONTRACT: top-level handle ids in the edge JSON are plain strings ('prompt', 'input_value', 'text', 'system_message'); rich metadata belongs only under data.sourceHandle / data.targetHandle.",
            "Never invent handle names. Only use handles present in registry.components for the referenced node types. If a handle is missing, choose another valid handle or insert an adapter.",
            "Examples:",
            "<TextInput:TextInput-Cy3bC: Prompt-3Um6W.out:prompt -> TextInput-Cy3bC.in:input_value, TextInput-Cy3bC.out:text -> LanguageModelComponent-lRD6W.in:system_message> inserts a TextInput between Prompt and LLM.",
            "<MessageToText:A1: TextInput-IN.out:text -> MessageToText-A1.in:message, MessageToText-A1.out:text -> Prompt-P1.in:input> inserts an adapter to convert Message→str, then feeds a Prompt variable.",
            "Use the official component names like PromptLayerOpenAI, ConversationSummaryBufferMemory, TextLoader, Chroma, OpenAIEmbeddings, GuardrailsComponent.",
            "ITS EXTREMELY IMPORTANT THAT BEFORE ADDING THE COMPONENT, YOU CRITICALLY LOOK AT registry.components AND CHECK each component’s inputTypes and output_types.",
            "CRITICAL: Use registry.components as the single source of truth for handle names and types; set top-level sourceHandle/targetHandle to the plain id strings; never guess or pad types—if output_types and inputTypes differ, insert the minimal adapter component and wire it explicitly.",
            "DATA TYPE CONTRACT: set data.sourceHandle.dataType to the source node’s component type (exact string from registry); data.targetHandle has no dataType—only {fieldName, inputTypes, type}; take output_types/inputTypes from registry; if they differ, insert an adapter (never pad or guess).",
            "registry.components:"
            "{'registry_version':'v2','generated_at':'2025-08-17T17:02:33.408541+00:00','root':'/Applications/XAMPP/xamppfiles/htdocs/aimable/langflow/index_payload/components','components':[{'type':'AIMLModel','display_name':'AI/ML API','inputs':{'max_tokens':[],'model_kwargs':[],'model_name':[],'aiml_api_base':[],'api_key':[],'temperature':[]},'outputs':{},'documentation':'https://docs.aimlapi.com/api-reference','source':'src/backend/base/langflow/components/aiml/aiml.py#L19'},{'type':'APIRequest','display_name':'API Request','inputs':{'url_input':[],'curl_input':[],'method':[],'mode':[],'query_params':[],'body':['Data'],'headers':['Data'],'timeout':[],'follow_redirects':[],'save_to_file':[],'include_httpx_metadata':[]},'outputs':{'data':[]},'documentation':'https://docs.langflow.org/components-data#api-request','source':'src/backend/base/langflow/components/data/api_request.py#L45'},{'type':'AddContentToPage','display_name':'Add Content to Page ','inputs':{'markdown_text':[],'block_id':[],'notion_secret':[]},'outputs':{},'documentation':'https://developers.notion.com/reference/patch-block-children','source':'src/backend/base/langflow/components/Notion/add_content_to_page.py#L19'},{'type':'Agent','display_name':'Agent','inputs':{'agent_llm':[],'system_prompt':[],'n_messages':[],'add_current_date_tool':[]},'outputs':{'response':[],'structured_response':[]},'documentation':'https://docs.langflow.org/agents','source':'src/backend/base/langflow/components/agents/agent.py#L37'},{'type':'AgentQL','display_name':'Extract Web Data','inputs':{'api_key':[],'url':[],'query':[],'prompt':[],'is_stealth_mode_enabled':[],'timeout':[],'mode':[],'wait_for':[],'is_scroll_to_bottom_enabled':[],'is_screenshot_enabled':[]},'outputs':{'data':[]},'documentation':'https://docs.agentql.com/rest-api/api-reference','source':'src/backend/base/langflow/components/agentql/agentql_api.py#L18'},{'type':'AlterMetadata','display_name':'Alter Metadata','inputs':{'input_value':['Message','Data'],'text_in':[],'metadata':['Data'],'remove_fields':[]},'outputs':{'data':[],'dataframe':[]},'source':'src/backend/base/langflow/components/processing/alter_metadata.py#L8'},{'type':'AmazonBedrockEmbeddings','display_name':'Amazon Bedrock Embeddings','inputs':{'model_id':[],'aws_access_key_id':[],'aws_secret_access_key':[],'aws_session_token':[],'credentials_profile_name':[],'region_name':[],'endpoint_url':[]},'outputs':{'embeddings':[]},'source':'src/backend/base/langflow/components/amazon/amazon_bedrock_embedding.py#L8'},{'type':'AmazonBedrockModel','display_name':'Amazon Bedrock','inputs':{'model_id':[],'aws_access_key_id':[],'aws_secret_access_key':[],'aws_session_token':[],'credentials_profile_name':[],'region_name':[],'model_kwargs':[],'endpoint_url':[]},'outputs':{},'source':'src/backend/base/langflow/components/amazon/amazon_bedrock_model.py#L8'},{'type':'AmazonKendra','display_name':'Amazon Kendra Retriever','inputs':{'index_id':[],'region_name':[],'credentials_profile_name':[],'attribute_filter':[],'top_k':[],'user_context':[]},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/amazon_kendra.py#L9'},{'type':'AnthropicModel','display_name':'Anthropic','inputs':{'max_tokens':[],'model_name':[],'api_key':[],'temperature':[],'base_url':[],'tool_model_enabled':[]},'outputs':{},'source':'src/backend/base/langflow/components/anthropic/anthropic.py#L20'},{'type':'ApifyActors','display_name':'Apify Actors','inputs':{'apify_token':[],'actor_id':[],'run_input':[],'dataset_fields':[],'flatten_dataset':[]},'outputs':{'output':[],'tool':[]},'documentation':'http://docs.langflow.org/integrations-apify','source':'src/backend/base/langflow/components/apify/apify_actor.py#L19'},{'type':'ArXivComponent','display_name':'arXiv','inputs':{'search_query':[],'search_type':[],'max_results':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/arxiv/arxiv.py#L13'},{'type':'AssemblyAIGetSubtitles','display_name':'AssemblyAI Get Subtitles','inputs':{'api_key':[],'transcription_result':[],'subtitle_format':[],'chars_per_caption':[]},'outputs':{'subtitles':[]},'documentation':'https://www.assemblyai.com/docs','source':'src/backend/base/langflow/components/assemblyai/assemblyai_get_subtitles.py#L9'},{'type':'AssemblyAILeMUR','display_name':'AssemblyAI LeMUR','inputs':{'api_key':[],'transcription_result':[],'prompt':[],'final_model':[],'temperature':[],'max_output_size':[],'endpoint':[],'questions':[],'transcript_ids':[]},'outputs':{'lemur_response':[]},'documentation':'https://www.assemblyai.com/docs/lemur','source':'src/backend/base/langflow/components/assemblyai/assemblyai_lemur.py#L9'},{'type':'AssemblyAIListTranscripts','display_name':'AssemblyAI List Transcripts','inputs':{'api_key':[],'limit':[],'status_filter':[],'created_on':[],'throttled_only':[]},'outputs':{'transcript_list':[]},'documentation':'https://www.assemblyai.com/docs','source':'src/backend/base/langflow/components/assemblyai/assemblyai_list_transcripts.py#L9'},{'type':'AssemblyAITranscriptionJobCreator','display_name':'AssemblyAI Start Transcript','inputs':{'api_key':[],'audio_file':[],'audio_file_url':[],'speech_model':[],'language_detection':[],'language_code':[],'speaker_labels':[],'speakers_expected':[],'punctuate':[],'format_text':[]},'outputs':{'transcript_id':[]},'documentation':'https://www.assemblyai.com/docs','source':'src/backend/base/langflow/components/assemblyai/assemblyai_start_transcript.py#L11'},{'type':'AssemblyAITranscriptionJobPoller','display_name':'AssemblyAI Poll Transcript','inputs':{'api_key':[],'transcript_id':[],'polling_interval':[]},'outputs':{'transcription_result':[]},'documentation':'https://www.assemblyai.com/docs','source':'src/backend/base/langflow/components/assemblyai/assemblyai_poll_transcript.py#L10'},{'type':'AstraDB','display_name':'Astra DB','inputs':{'token':[],'environment':[],'database_name':[],'api_endpoint':[],'keyspace':[],'collection_name':[],'embedding_model':['Embeddings'],'search_method':[],'reranker':[],'lexical_terms':[],'number_of_results':[],'search_type':[],'search_score_threshold':[],'advanced_search_filter':[],'autodetect_collection':[],'content_field':[],'deletion_field':[],'ignore_invalid_documents':[],'astradb_vectorstore_kwargs':[]},'outputs':{},'documentation':'https://docs.datastax.com/en/langflow/astra-components.html','source':'src/backend/base/langflow/components/vectorstores/astradb.py#L31'},{'type':'AstraDBCQLToolComponent','display_name':'Astra DB CQL','inputs':{'tool_name':[],'tool_description':[],'keyspace':[],'table_name':[],'token':[],'api_endpoint':[],'projection_fields':[],'tools_params':[],'partition_keys':[],'clustering_keys':[],'static_filters':[],'number_of_results':[]},'outputs':{},'documentation':'https://docs.langflow.org/Components/components-tools#astra-db-cql-tool','source':'src/backend/base/langflow/components/datastax/astradb_cql.py#L18'},{'type':'AstraDBChatMemory','display_name':'Astra DB Chat Memory','inputs':{'token':[],'api_endpoint':[],'collection_name':[],'namespace':[],'session_id':[]},'outputs':{},'source':'src/backend/base/langflow/components/datastax/astra_db.py#L10'},{'type':'AstraDBGraph','display_name':'Astra DB Graph','inputs':{'token':[],'api_endpoint':[],'collection_name':[],'metadata_incoming_links_key':[],'keyspace':[],'embedding_model':['Embeddings'],'metric':[],'batch_size':[],'bulk_insert_batch_concurrency':[],'bulk_insert_overwrite_concurrency':[],'bulk_delete_concurrency':[],'setup_mode':[],'pre_delete_collection':[],'metadata_indexing_include':[],'metadata_indexing_exclude':[],'collection_indexing_policy':[],'number_of_results':[],'search_type':[],'search_score_threshold':[],'search_filter':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/astradb_graph.py#L21'},{'type':'AstraDBToolComponent','display_name':'Astra DB Tool','inputs':{'tool_name':[],'tool_description':[],'keyspace':[],'collection_name':[],'token':[],'api_endpoint':[],'projection_attributes':[],'tools_params_v2':[],'tool_params':[],'static_filters':[],'number_of_results':[],'use_search_query':[],'use_vectorize':[],'embedding':['Embeddings'],'semantic_search_instruction':[]},'outputs':{},'documentation':'https://docs.langflow.org/components-bundle-components','source':'src/backend/base/langflow/components/datastax/astradb_tool.py#L17'},{'type':'AstraVectorize','display_name':'Astra Vectorize [DEPRECATED]','inputs':{'provider':[],'model_name':[],'api_key_name':[],'authentication':[],'provider_api_key':[],'model_parameters':[]},'outputs':{'config':['dict']},'documentation':'https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html','source':'src/backend/base/langflow/components/datastax/astra_vectorize.py#L8'},{'type':'AzureOpenAIEmbeddings','display_name':'Azure OpenAI Embeddings','inputs':{'model':[],'azure_endpoint':[],'azure_deployment':[],'api_version':[],'api_key':[],'dimensions':[]},'outputs':{'embeddings':[]},'documentation':'https://python.langchain.com/docs/integrations/text_embedding/azureopenai','source':'src/backend/base/langflow/components/azure/azure_openai_embeddings.py#L9'},{'type':'AzureOpenAIModel','display_name':'Azure OpenAI','inputs':{'azure_endpoint':[],'azure_deployment':[],'api_key':[],'api_version':[],'temperature':[],'max_tokens':[]},'outputs':{},'documentation':'https://python.langchain.com/docs/integrations/llms/azure_openai','source':'src/backend/base/langflow/components/azure/azure_openai.py#L10'},{'type':'BaiduQianfanChatModel','display_name':'Qianfan','inputs':{'model':[],'qianfan_ak':[],'qianfan_sk':[],'top_p':[],'temperature':[],'penalty_score':[],'endpoint':[]},'outputs':{},'documentation':'https://python.langchain.com/docs/integrations/chat/baidu_qianfan_endpoint','source':'src/backend/base/langflow/components/baidu/baidu_qianfan_chat.py#L8'},{'type':'BatchRunComponent','display_name':'Batch Run','inputs':{'model':['LanguageModel'],'system_message':[],'df':[],'column_name':[],'output_column_name':[],'enable_metadata':[]},'outputs':{'batch_results':[]},'documentation':'https://docs.langflow.org/components-processing#batch-run','source':'src/backend/base/langflow/components/processing/batch_run.py#L16'},{'type':'BigQueryExecutor','display_name':'BigQuery','inputs':{'service_account_json_file':[],'query':[],'clean_query':[]},'outputs':{'query_results':[]},'source':'src/backend/base/langflow/components/google/google_bq_sql_executor.py#L14'},{'type':'BingSearchAPI','display_name':'Bing Search API','inputs':{'bing_subscription_key':[],'input_value':[],'bing_search_url':[],'k':[]},'outputs':{'dataframe':[],'tool':[]},'source':'src/backend/base/langflow/components/bing/bing_search_api.py#L14'},{'type':'CSVAgent','display_name':'CSV Agent','inputs':{'llm':['LanguageModel'],'path':['str','Message'],'agent_type':[],'input_value':[],'pandas_kwargs':[]},'outputs':{'response':[],'agent':[]},'documentation':'https://python.langchain.com/docs/modules/agents/toolkits/csv','source':'src/backend/base/langflow/components/langchain_utilities/csv_agent.py#L16'},{'type':'CSVtoData','display_name':'Load CSV','inputs':{'csv_file':[],'csv_path':[],'csv_string':[],'text_key':[]},'outputs':{'data_list':[]},'source':'src/backend/base/langflow/components/data/csv_to_data.py#L10'},{'type':'CalculatorComponent','display_name':'Calculator','inputs':{'expression':[]},'outputs':{'result':[]},'documentation':'https://docs.langflow.org/components-helpers#calculator','source':'src/backend/base/langflow/components/helpers/calculator_core.py#L11'},{'type':'CalculatorTool','display_name':'Calculator [DEPRECATED]','inputs':{'expression':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/calculator.py#L15'},{'type':'Cassandra','display_name':'Cassandra','inputs':{'database_ref':[],'username':[],'token':[],'keyspace':[],'table_name':[],'ttl_seconds':[],'batch_size':[],'setup_mode':[],'cluster_kwargs':[],'embedding':['Embeddings'],'number_of_results':[],'search_type':[],'search_score_threshold':[],'search_filter':[],'body_search':[],'enable_body_search':[]},'outputs':{},'documentation':'https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/cassandra','source':'src/backend/base/langflow/components/vectorstores/cassandra.py#L16'},{'type':'CassandraChatMemory','display_name':'Cassandra Chat Memory','inputs':{'database_ref':[],'username':[],'token':[],'keyspace':[],'table_name':[],'session_id':[],'cluster_kwargs':[]},'outputs':{},'source':'src/backend/base/langflow/components/datastax/cassandra.py#L6'},{'type':'CassandraGraph','display_name':'Cassandra Graph','inputs':{'database_ref':[],'username':[],'token':[],'keyspace':[],'table_name':[],'setup_mode':[],'cluster_kwargs':[],'embedding':['Embeddings'],'number_of_results':[],'search_type':[],'depth':[],'search_score_threshold':[],'search_filter':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/cassandra_graph.py#L18'},{'type':'CharacterTextSplitter','display_name':'Character Text Splitter','inputs':{'chunk_size':[],'chunk_overlap':[],'data_input':['Document','Data'],'separator':[]},'outputs':{},'documentation':'https://docs.langflow.org/components/text-splitters#charactertextsplitter','source':'src/backend/base/langflow/components/langchain_utilities/character.py#L10'},{'type':'ChatInput','display_name':'Chat Input','inputs':{'input_value':[],'should_store_message':[],'sender':[],'sender_name':[],'session_id':[],'files':[],'background_color':[],'chat_icon':[],'text_color':[]},'outputs':{'message':[]},'documentation':'https://docs.langflow.org/components-io#chat-input','source':'src/backend/base/langflow/components/input_output/chat.py#L19'},{'type':'ChatLiteLLMModelComponent','display_name':'LiteLLM','inputs':{'input_value':[],'model':[],'api_key':[],'provider':[],'temperature':[],'kwargs':[],'model_kwargs':[],'top_p':[],'top_k':[],'n':[],'max_tokens':[],'max_retries':[],'verbose':[],'stream':[],'system_message':[]},'outputs':{},'documentation':'https://python.langchain.com/docs/integrations/chat/litellm','source':'src/backend/base/langflow/components/deactivated/chat_litellm_model.py#L18'},{'type':'ChatOutput','display_name':'Chat Output','inputs':{'input_value':['Data','DataFrame','Message'],'should_store_message':[],'sender':[],'sender_name':[],'session_id':[],'data_template':[],'background_color':[],'chat_icon':[],'text_color':[],'clean_data':[]},'outputs':{'message':[]},'documentation':'https://docs.langflow.org/components-io#chat-output','source':'src/backend/base/langflow/components/input_output/chat_output.py#L22'},{'type':'Chroma','display_name':'Chroma DB','inputs':{'collection_name':[],'persist_directory':[],'embedding':['Embeddings'],'chroma_server_cors_allow_origins':[],'chroma_server_host':[],'chroma_server_http_port':[],'chroma_server_grpc_port':[],'chroma_server_ssl_enabled':[],'allow_duplicates':[],'search_type':[],'number_of_results':[],'limit':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/chroma.py#L17'},{'type':'ChunkDoclingDocument','display_name':'Chunk DoclingDocument','inputs':{'data_inputs':['Data','DataFrame'],'chunker':[],'provider':[],'hf_model_name':[],'openai_model_name':[],'max_tokens':[],'doc_key':[]},'outputs':{'dataframe':[]},'documentation':'https://docling-project.github.io/docling/concepts/chunking/','source':'src/backend/base/langflow/components/docling/chunk_docling_document.py#L13'},{'type':'CleanlabEvaluator','display_name':'Cleanlab Evaluator','inputs':{'system_prompt':[],'prompt':[],'response':[],'api_key':[],'model':[],'quality_preset':[]},'outputs':{'response_passthrough':['Message'],'score':['number'],'explanation':['Message']},'source':'src/backend/base/langflow/components/cleanlab/cleanlab_evaluator.py#L13'},{'type':'CleanlabRAGEvaluator','display_name':'Cleanlab RAG Evaluator','inputs':{'api_key':[],'model':[],'quality_preset':[],'context':[],'query':[],'response':[],'run_context_sufficiency':[],'run_response_groundedness':[],'run_response_helpfulness':[],'run_query_ease':[]},'outputs':{'response_passthrough':['Message'],'trust_score':['number'],'trust_explanation':['Message'],'other_scores':['Data'],'evaluation_summary':['Message']},'source':'src/backend/base/langflow/components/cleanlab/cleanlab_rag_evaluator.py#L14'},{'type':'CleanlabRemediator','display_name':'Cleanlab Remediator','inputs':{'response':[],'score':['number'],'explanation':[],'threshold':[],'show_untrustworthy_response':[],'untrustworthy_warning_text':[],'fallback_text':[]},'outputs':{'remediated_response':['Message']},'source':'src/backend/base/langflow/components/cleanlab/cleanlab_remediator.py#L7'},{'type':'Clickhouse','display_name':'Clickhouse','inputs':{'host':[],'port':[],'database':[],'table':[],'username':[],'password':[],'index_type':[],'metric':[],'secure':[],'index_param':[],'index_query_params':[],'embedding':['Embeddings'],'number_of_results':[],'score_threshold':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/clickhouse.py#L17'},{'type':'CloudflareWorkersAIEmbeddings','display_name':'Cloudflare Workers AI Embeddings','inputs':{'account_id':[],'api_token':[],'model_name':[],'strip_new_lines':[],'batch_size':[],'api_base_url':[],'headers':[]},'outputs':{'embeddings':[]},'documentation':'https://python.langchain.com/docs/integrations/text_embedding/cloudflare_workersai/','source':'src/backend/base/langflow/components/cloudflare/cloudflare.py#L8'},{'type':'CodeBlockExtractor','display_name':'Code Block Extractor','inputs':{'text':[]},'outputs':{'code_block':[]},'source':'src/backend/base/langflow/components/deactivated/code_block_extractor.py#L7'},{'type':'CohereEmbeddings','display_name':'Cohere Embeddings','inputs':{'api_key':[],'model_name':[],'truncate':[],'max_retries':[],'user_agent':[],'request_timeout':[]},'outputs':{'embeddings':[]},'source':'src/backend/base/langflow/components/cohere/cohere_embeddings.py#L13'},{'type':'CohereModel','display_name':'Cohere Language Models','inputs':{'cohere_api_key':[],'temperature':[]},'outputs':{},'documentation':'https://python.langchain.com/docs/modules/model_io/models/llms/integrations/cohere','source':'src/backend/base/langflow/components/cohere/cohere_models.py#L10'},{'type':'CohereRerank','display_name':'Cohere Rerank','inputs':{'api_key':[],'model':[]},'outputs':{'reranked_documents':[]},'source':'src/backend/base/langflow/components/cohere/cohere_rerank.py#L8'},{'type':'Combinatorial Reasoner','display_name':'Combinatorial Reasoner','inputs':{'prompt':[],'openai_api_key':[],'username':[],'password':[],'model_name':[]},'outputs':{'optimized_prompt':[],'reasons':[]},'source':'src/backend/base/langflow/components/icosacomputing/combinatorial_reasoner.py#L12'},{'type':'CombineText','display_name':'Combine Text','inputs':{'text1':[],'text2':[],'delimiter':[]},'outputs':{'combined_text':[]},'source':'src/backend/base/langflow/components/processing/combine_text.py#L6'},{'type':'ComposioAPI','display_name':'Composio Tools','inputs':{'entity_id':[],'api_key':[],'tool_name':[],'actions':[]},'outputs':{'tools':[]},'documentation':'https://docs.composio.dev','source':'src/backend/base/langflow/components/composio/composio_api.py#L25'},{'type':'ComposioGitHubAPIComponent','display_name':'GitHub','inputs':{'GITHUB_CREATE_AN_ISSUE_owner':[],'GITHUB_CREATE_AN_ISSUE_repo':[],'GITHUB_CREATE_AN_ISSUE_title':[],'GITHUB_CREATE_AN_ISSUE_body':[],'GITHUB_CREATE_AN_ISSUE_assignee':[],'GITHUB_CREATE_AN_ISSUE_milestone':[],'GITHUB_CREATE_AN_ISSUE_labels':[],'GITHUB_CREATE_AN_ISSUE_assignees':[],'GITHUB_LIST_PULL_REQUESTS_owner':[],'GITHUB_LIST_PULL_REQUESTS_repo':[],'GITHUB_LIST_PULL_REQUESTS_state':[],'GITHUB_LIST_PULL_REQUESTS_head':[],'GITHUB_LIST_PULL_REQUESTS_base':[],'GITHUB_LIST_PULL_REQUESTS_sort':[],'GITHUB_LIST_PULL_REQUESTS_direction':[],'GITHUB_LIST_PULL_REQUESTS_per_page':[],'GITHUB_LIST_PULL_REQUESTS_page':[],'GITHUB_CREATE_A_PULL_REQUEST_owner':[],'GITHUB_CREATE_A_PULL_REQUEST_repo':[],'GITHUB_CREATE_A_PULL_REQUEST_title':[],'GITHUB_CREATE_A_PULL_REQUEST_head':[],'GITHUB_CREATE_A_PULL_REQUEST_head_repo':[],'GITHUB_CREATE_A_PULL_REQUEST_base':[],'GITHUB_CREATE_A_PULL_REQUEST_body':[],'GITHUB_CREATE_A_PULL_REQUEST_maintainer_can_modify':[],'GITHUB_CREATE_A_PULL_REQUEST_draft':[],'GITHUB_CREATE_A_PULL_REQUEST_issue':[],'GITHUB_LIST_REPOSITORY_ISSUES_owner':[],'GITHUB_LIST_REPOSITORY_ISSUES_repo':[],'GITHUB_LIST_REPOSITORY_ISSUES_milestone':[],'GITHUB_LIST_REPOSITORY_ISSUES_state':[],'GITHUB_LIST_REPOSITORY_ISSUES_assignee':[],'GITHUB_LIST_REPOSITORY_ISSUES_creator':[],'GITHUB_LIST_REPOSITORY_ISSUES_mentioned':[],'GITHUB_LIST_REPOSITORY_ISSUES_labels':[],'GITHUB_LIST_REPOSITORY_ISSUES_sort':[],'GITHUB_LIST_REPOSITORY_ISSUES_direction':[],'GITHUB_LIST_REPOSITORY_ISSUES_since':[],'GITHUB_LIST_REPOSITORY_ISSUES_per_page':[],'GITHUB_LIST_REPOSITORY_ISSUES_page':[],'GITHUB_LIST_BRANCHES_owner':[],'GITHUB_LIST_BRANCHES_repo':[],'GITHUB_LIST_BRANCHES_protected':[],'GITHUB_LIST_BRANCHES_per_page':[],'GITHUB_LIST_BRANCHES_page':[],'GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER_owner':[],'GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER_repo':[],'GITHUB_GET_A_PULL_REQUEST_owner':[],'GITHUB_GET_A_PULL_REQUEST_repo':[],'GITHUB_GET_A_PULL_REQUEST_pull_number':[],'GITHUB_LIST_COMMITS_owner':[],'GITHUB_LIST_COMMITS_repo':[],'GITHUB_LIST_COMMITS_sha':[],'GITHUB_LIST_COMMITS_path':[],'GITHUB_LIST_COMMITS_author':[],'GITHUB_LIST_COMMITS_committer':[],'GITHUB_LIST_COMMITS_since':[],'GITHUB_LIST_COMMITS_until':[],'GITHUB_LIST_COMMITS_per_page':[],'GITHUB_LIST_COMMITS_page':[]},'outputs':{},'documentation':'https://docs.composio.dev','source':'src/backend/base/langflow/components/composio/github_composio.py#L15'},{'type':'ComposioGoogleCalendarAPIComponent','display_name':'Google Calendar','inputs':{'GOOGLECALENDAR_LIST_CALENDARS_max_results':[],'GOOGLECALENDAR_LIST_CALENDARS_min_access_role':[],'GOOGLECALENDAR_LIST_CALENDARS_page_token':[],'GOOGLECALENDAR_LIST_CALENDARS_show_deleted':[],'GOOGLECALENDAR_LIST_CALENDARS_show_hidden':[],'GOOGLECALENDAR_LIST_CALENDARS_sync_token':[],'GOOGLECALENDAR_FIND_EVENT_calendar_id':[],'GOOGLECALENDAR_FIND_EVENT_query':[],'GOOGLECALENDAR_FIND_EVENT_max_results':[],'GOOGLECALENDAR_FIND_EVENT_order_by':[],'GOOGLECALENDAR_FIND_EVENT_show_deleted':[],'GOOGLECALENDAR_FIND_EVENT_single_events':[],'GOOGLECALENDAR_FIND_EVENT_timeMax':[],'GOOGLECALENDAR_FIND_EVENT_timeMin':[],'GOOGLECALENDAR_FIND_EVENT_updated_min':[],'GOOGLECALENDAR_FIND_EVENT_event_types':[],'GOOGLECALENDAR_FIND_EVENT_page_token':[],'GOOGLECALENDAR_DUPLICATE_CALENDAR_summary':[],'GOOGLECALENDAR_REMOVE_ATTENDEE_calendar_id':[],'GOOGLECALENDAR_REMOVE_ATTENDEE_event_id':[],'GOOGLECALENDAR_REMOVE_ATTENDEE_attendee_email':[],'GOOGLECALENDAR_GET_CALENDAR_calendar_id':[],'GOOGLECALENDAR_CREATE_EVENT_description':[],'GOOGLECALENDAR_CREATE_EVENT_eventType':[],'GOOGLECALENDAR_CREATE_EVENT_create_meeting_room':[],'GOOGLECALENDAR_CREATE_EVENT_guestsCanSeeOtherGuests':[],'GOOGLECALENDAR_CREATE_EVENT_guestsCanInviteOthers':[],'GOOGLECALENDAR_CREATE_EVENT_location':[],'GOOGLECALENDAR_CREATE_EVENT_summary':[],'GOOGLECALENDAR_CREATE_EVENT_transparency':[],'GOOGLECALENDAR_CREATE_EVENT_visibility':[],'GOOGLECALENDAR_CREATE_EVENT_timezone':[],'GOOGLECALENDAR_CREATE_EVENT_recurrence':[],'GOOGLECALENDAR_CREATE_EVENT_guests_can_modify':[],'GOOGLECALENDAR_CREATE_EVENT_attendees':[],'GOOGLECALENDAR_CREATE_EVENT_send_updates':[],'GOOGLECALENDAR_CREATE_EVENT_start_datetime':[],'GOOGLECALENDAR_CREATE_EVENT_event_duration_hour':[],'GOOGLECALENDAR_CREATE_EVENT_event_duration_minutes':[],'GOOGLECALENDAR_CREATE_EVENT_calendar_id':[],'GOOGLECALENDAR_DELETE_EVENT_calendar_id':[],'GOOGLECALENDAR_DELETE_EVENT_event_id':[],'GOOGLECALENDAR_FIND_FREE_SLOTS_time_min':[],'GOOGLECALENDAR_FIND_FREE_SLOTS_time_max':[],'GOOGLECALENDAR_FIND_FREE_SLOTS_timezone':[],'GOOGLECALENDAR_FIND_FREE_SLOTS_group_expansion_max':[],'GOOGLECALENDAR_FIND_FREE_SLOTS_calendar_expansion_max':[],'GOOGLECALENDAR_FIND_FREE_SLOTS_items':[],'GOOGLECALENDAR_QUICK_ADD_calendar_id':[],'GOOGLECALENDAR_QUICK_ADD_text':[],'GOOGLECALENDAR_QUICK_ADD_send_updates':[],'GOOGLECALENDAR_PATCH_CALENDAR_calendar_id':[],'GOOGLECALENDAR_PATCH_CALENDAR_description':[],'GOOGLECALENDAR_PATCH_CALENDAR_location':[],'GOOGLECALENDAR_PATCH_CALENDAR_summary':[],'GOOGLECALENDAR_PATCH_CALENDAR_timezone':[],'GOOGLECALENDAR_GET_CURRENT_DATE_TIME_timezone':[],'GOOGLECALENDAR_UPDATE_EVENT_description':[],'GOOGLECALENDAR_UPDATE_EVENT_eventType':[],'GOOGLECALENDAR_UPDATE_EVENT_create_meeting_room':[],'GOOGLECALENDAR_UPDATE_EVENT_guestsCanSeeOtherGuests':[],'GOOGLECALENDAR_UPDATE_EVENT_guestsCanInviteOthers':[],'GOOGLECALENDAR_UPDATE_EVENT_location':[],'GOOGLECALENDAR_UPDATE_EVENT_summary':[],'GOOGLECALENDAR_UPDATE_EVENT_transparency':[],'GOOGLECALENDAR_UPDATE_EVENT_visibility':[],'GOOGLECALENDAR_UPDATE_EVENT_timezone':[],'GOOGLECALENDAR_UPDATE_EVENT_recurrence':[],'GOOGLECALENDAR_UPDATE_EVENT_guests_can_modify':[],'GOOGLECALENDAR_UPDATE_EVENT_attendees':[],'GOOGLECALENDAR_UPDATE_EVENT_send_updates':[],'GOOGLECALENDAR_UPDATE_EVENT_start_datetime':[],'GOOGLECALENDAR_UPDATE_EVENT_event_duration_hour':[],'GOOGLECALENDAR_UPDATE_EVENT_event_duration_minutes':[],'GOOGLECALENDAR_UPDATE_EVENT_calendar_id':[],'GOOGLECALENDAR_UPDATE_EVENT_event_id':[]},'outputs':{},'documentation':'https://docs.composio.dev','source':'src/backend/base/langflow/components/composio/googlecalendar_composio.py#L14'},{'type':'ComposioOutlookAPIComponent','display_name':'Outlook','inputs':{'OUTLOOK_OUTLOOK_LIST_EVENTS_user_id':[],'OUTLOOK_OUTLOOK_LIST_EVENTS_top':[],'OUTLOOK_OUTLOOK_LIST_EVENTS_skip':[],'OUTLOOK_OUTLOOK_LIST_EVENTS_filter':[],'OUTLOOK_OUTLOOK_LIST_EVENTS_select':[],'OUTLOOK_OUTLOOK_LIST_EVENTS_orderby':[],'OUTLOOK_OUTLOOK_LIST_EVENTS_timezone':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_user_id':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_subject':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_body':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_to_email':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_to_name':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_cc_emails':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_bcc_emails':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_is_html':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_save_to_sent_items':[],'OUTLOOK_OUTLOOK_SEND_EMAIL_attachment':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_subject':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_body':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_to_recipients':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_cc_recipients':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_bcc_recipients':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_is_html':[],'OUTLOOK_OUTLOOK_CREATE_DRAFT_attachment':[],'OUTLOOK_OUTLOOK_GET_PROFILE_user_id':[],'OUTLOOK_OUTLOOK_REPLY_EMAIL_user_id':[],'OUTLOOK_OUTLOOK_REPLY_EMAIL_message_id':[],'OUTLOOK_OUTLOOK_REPLY_EMAIL_comment':[],'OUTLOOK_OUTLOOK_REPLY_EMAIL_cc_emails':[],'OUTLOOK_OUTLOOK_REPLY_EMAIL_bcc_emails':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_user_id':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_subject':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_body':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_is_html':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_start_datetime':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_end_datetime':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_time_zone':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_is_online_meeting':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_online_meeting_provider':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_attendees_info':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_location':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_show_as':[],'OUTLOOK_OUTLOOK_CALENDAR_CREATE_EVENT_categories':[],'OUTLOOK_OUTLOOK_GET_EVENT_user_id':[],'OUTLOOK_OUTLOOK_GET_EVENT_event_id':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_user_id':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_folder':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_top':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_skip':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_is_read':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_importance':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_subject':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_received_date_time_gt':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_subject_startswith':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_subject_endswith':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_subject_contains':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_received_date_time_ge':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_received_date_time_lt':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_received_date_time_le':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_from_address':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_has_attachments':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_body_preview_contains':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_sent_date_time_gt':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_sent_date_time_lt':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_categories':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_select':[],'OUTLOOK_OUTLOOK_LIST_MESSAGES_orderby':[]},'outputs':{},'documentation':'https://docs.composio.dev','source':'src/backend/base/langflow/components/composio/outlook_composio.py#L11'},{'type':'ComposioSlackAPIComponent','display_name':'Slack','inputs':{'SLACK_LIST_ALL_SLACK_TEAM_USERS_WITH_PAGINATION_limit':[],'SLACK_LIST_ALL_SLACK_TEAM_USERS_WITH_PAGINATION_cursor':[],'SLACK_LIST_ALL_SLACK_TEAM_USERS_WITH_PAGINATION_include_locale':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_as_user':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_attachments':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_blocks':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_channel':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_icon_emoji':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_icon_url':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_link_names':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_mrkdwn':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_parse':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_reply_broadcast':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_text':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_thread_ts':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_unfurl_links':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_unfurl_media':[],'SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL_username':[],'SLACK_UPDATES_A_SLACK_MESSAGE_as_user':[],'SLACK_UPDATES_A_SLACK_MESSAGE_attachments':[],'SLACK_UPDATES_A_SLACK_MESSAGE_blocks':[],'SLACK_UPDATES_A_SLACK_MESSAGE_channel':[],'SLACK_UPDATES_A_SLACK_MESSAGE_link_names':[],'SLACK_UPDATES_A_SLACK_MESSAGE_parse':[],'SLACK_UPDATES_A_SLACK_MESSAGE_text':[],'SLACK_UPDATES_A_SLACK_MESSAGE_ts':[],'SLACK_FETCH_CONVERSATION_HISTORY_channel':[],'SLACK_FETCH_CONVERSATION_HISTORY_latest':[],'SLACK_FETCH_CONVERSATION_HISTORY_oldest':[],'SLACK_FETCH_CONVERSATION_HISTORY_inclusive':[],'SLACK_FETCH_CONVERSATION_HISTORY_limit':[],'SLACK_FETCH_CONVERSATION_HISTORY_cursor':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_as_user':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_attachments':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_blocks':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_channel':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_link_names':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_parse':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_post_at':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_reply_broadcast':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_text':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_thread_ts':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_unfurl_links':[],'SLACK_SCHEDULES_A_MESSAGE_TO_A_CHANNEL_AT_A_SPECIFIED_TIME_unfurl_media':[],'SLACK_LIST_ALL_SLACK_TEAM_CHANNELS_WITH_VARIOUS_FILTERS_exclude_archived':[],'SLACK_LIST_ALL_SLACK_TEAM_CHANNELS_WITH_VARIOUS_FILTERS_types':[],'SLACK_LIST_ALL_SLACK_TEAM_CHANNELS_WITH_VARIOUS_FILTERS_limit':[],'SLACK_LIST_ALL_SLACK_TEAM_CHANNELS_WITH_VARIOUS_FILTERS_cursor':[],'SLACK_SEARCH_FOR_MESSAGES_WITH_QUERY_count':[],'SLACK_SEARCH_FOR_MESSAGES_WITH_QUERY_highlight':[],'SLACK_SEARCH_FOR_MESSAGES_WITH_QUERY_page':[],'SLACK_SEARCH_FOR_MESSAGES_WITH_QUERY_query':[],'SLACK_SEARCH_FOR_MESSAGES_WITH_QUERY_sort':[],'SLACK_SEARCH_FOR_MESSAGES_WITH_QUERY_sort_dir':[],'SLACK_CREATE_A_REMINDER_text':[],'SLACK_CREATE_A_REMINDER_time':[],'SLACK_CREATE_A_REMINDER_user':[]},'outputs':{},'documentation':'https://docs.composio.dev','source':'src/backend/base/langflow/components/composio/slack_composio.py#L14'},{'type':'ConditionalRouter','display_name':'If-Else','inputs':{'input_text':[],'operator':[],'match_text':[],'case_sensitive':[],'true_case_message':[],'false_case_message':[],'max_iterations':[],'default_route':[]},'outputs':{'true_result':[],'false_result':[]},'documentation':'https://docs.langflow.org/components-logic#conditional-router-if-else-component','source':'src/backend/base/langflow/components/logic/conditional_router.py#L8'},{'type':'Confluence','display_name':'Confluence','inputs':{'url':[],'username':[],'api_key':[],'space_key':[],'cloud':[],'content_format':[],'max_pages':[]},'outputs':{'data':[]},'documentation':'https://python.langchain.com/v0.2/docs/integrations/document_loaders/confluence/','source':'src/backend/base/langflow/components/confluence/confluence.py#L9'},{'type':'ConversationChain','display_name':'ConversationChain','inputs':{'input_value':[],'llm':['LanguageModel'],'memory':['BaseChatMemory']},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/conversation.py#L8'},{'type':'ConvertAstraToTwelveLabs','display_name':'Convert Astra DB to Pegasus Input','inputs':{'astra_results':['Data']},'outputs':{'index_id':[],'video_id':[]},'documentation':'https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md','source':'src/backend/base/langflow/components/twelvelabs/convert_astra_results.py#L9'},{'type':'Couchbase','display_name':'Couchbase','inputs':{'couchbase_connection_string':[],'couchbase_username':[],'couchbase_password':[],'bucket_name':[],'scope_name':[],'collection_name':[],'index_name':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/couchbase.py#L11'},{'type':'CreateData','display_name':'Create Data','inputs':{'number_of_fields':[],'text_key':[],'text_key_validator':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/processing/create_data.py#L11'},{'type':'CreateList','display_name':'Create List','inputs':{'texts':[]},'outputs':{'list':[],'dataframe':[]},'source':'src/backend/base/langflow/components/helpers/create_list.py#L8'},{'type':'CrewAIAgentComponent','display_name':'CrewAI Agent','inputs':{'role':[],'goal':[],'backstory':[],'tools':['Tool'],'llm':['LanguageModel'],'memory':[],'verbose':[],'allow_delegation':[],'allow_code_execution':[],'kwargs':[]},'outputs':{'output':[]},'documentation':'https://docs.crewai.com/how-to/LLM-Connections/','source':'src/backend/base/langflow/components/crewai/crewai.py#L6'},{'type':'CurrentDate','display_name':'Current Date','inputs':{'timezone':[]},'outputs':{'current_date':[]},'documentation':'https://docs.langflow.org/components-helpers#current-date','source':'src/backend/base/langflow/components/helpers/current_date.py#L11'},{'type':'CustomComponent','display_name':'Custom Component','inputs':{'input_value':[]},'outputs':{'output':[]},'documentation':'https://docs.langflow.org/components-custom-components','source':'src/backend/base/langflow/components/custom_component/custom_component.py#L7'},{'type':'DataConditionalRouter','display_name':'Condition','inputs':{'data_input':[],'key_name':[],'operator':[],'compare_value':[]},'outputs':{'true_output':[],'false_output':[]},'source':'src/backend/base/langflow/components/logic/data_conditional_router.py#L9'},{'type':'DataFrameOperations','display_name':'DataFrame Operations','inputs':{'df':[],'operation':[],'column_name':[],'filter_value':[],'filter_operator':[],'ascending':[],'new_column_name':[],'new_column_value':[],'columns_to_select':[],'num_rows':[],'replace_value':[],'replacement_value':[]},'outputs':{'output':[]},'documentation':'https://docs.langflow.org/components-processing#dataframe-operations','source':'src/backend/base/langflow/components/processing/dataframe_operations.py#L18'},{'type':'DataOperations','display_name':'Data Operations','inputs':{'data':[],'operations':[],'select_keys_input':[],'filter_key':[],'operator':[],'filter_values':[],'append_update_data':[],'remove_keys_input':[],'rename_keys_input':[]},'outputs':{'data_output':[]},'documentation':'https://docs.langflow.org/components-processing#data-operations','source':'src/backend/base/langflow/components/processing/data_operations.py#L33'},{'type':'DataToDataFrame','display_name':'Data → DataFrame','inputs':{'data_list':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/processing/data_to_dataframe.py#L7'},{'type':'DeepSeekModelComponent','display_name':'DeepSeek','inputs':{'max_tokens':[],'model_kwargs':[],'json_mode':[],'model_name':[],'api_base':[],'api_key':[],'temperature':[],'seed':[]},'outputs':{},'source':'src/backend/base/langflow/components/deepseek/deepseek.py#L13'},{'type':'Directory','display_name':'Directory','inputs':{'path':[],'types':[],'depth':[],'max_concurrency':[],'load_hidden':[],'recursive':[],'silent_errors':[],'use_multithreading':[]},'outputs':{'dataframe':[]},'documentation':'https://docs.langflow.org/components-data#directory','source':'src/backend/base/langflow/components/data/directory.py#L9'},{'type':'DoclingInline','display_name':'Docling','inputs':{'pipeline':[],'ocr_engine':[]},'outputs':{},'documentation':'https://docling-project.github.io/docling/','source':'src/backend/base/langflow/components/docling/docling_inline.py#L6'},{'type':'DoclingRemote','display_name':'Docling Serve','inputs':{'api_url':[],'max_concurrency':[],'max_poll_timeout':[],'api_headers':[],'docling_serve_opts':[]},'outputs':{},'documentation':'https://docling-project.github.io/docling/','source':'src/backend/base/langflow/components/docling/docling_remote.py#L17'},{'type':'DocumentsToData','display_name':'Documents ⇢ Data','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/documents_to_data.py#L7'},{'type':'Dotenv','display_name':'Dotenv','inputs':{'dotenv_file_content':[]},'outputs':{'env_set':[]},'source':'src/backend/base/langflow/components/datastax/dotenv.py#L11'},{'type':'DuckDuckGoSearchComponent','display_name':'DuckDuckGo Search','inputs':{'input_value':[],'max_results':[],'max_snippet_length':[]},'outputs':{'dataframe':[]},'documentation':'https://python.langchain.com/docs/integrations/tools/ddg','source':'src/backend/base/langflow/components/duckduckgo/duck_duck_go_search_run.py#L10'},{'type':'Elasticsearch','display_name':'Elasticsearch','inputs':{'elasticsearch_url':[],'cloud_id':[],'index_name':[],'username':[],'password':[],'embedding':['Embeddings'],'search_type':[],'number_of_results':[],'search_score_threshold':[],'api_key':[],'verify_certs':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/elasticsearch.py#L20'},{'type':'Embed','display_name':'Embed Texts','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/embed.py#L6'},{'type':'EmbeddingSimilarityComponent','display_name':'Embedding Similarity','inputs':{'embedding_vectors':[],'similarity_metric':[]},'outputs':{'similarity_data':[]},'source':'src/backend/base/langflow/components/embeddings/similarity.py#L10'},{'type':'ExaSearch','display_name':'Exa Search','inputs':{'metaphor_api_key':[],'use_autoprompt':[],'search_num_results':[],'similar_num_results':[]},'outputs':{'tools':[]},'documentation':'https://python.langchain.com/docs/integrations/tools/metaphor_search','source':'src/backend/base/langflow/components/exa/exa_search.py#L9'},{'type':'ExportDoclingDocument','display_name':'Export DoclingDocument','inputs':{'data_inputs':['Data','DataFrame'],'export_format':[],'image_mode':[],'md_image_placeholder':[],'md_page_break_placeholder':[],'doc_key':[]},'outputs':{'data':[],'dataframe':[]},'documentation':'https://docling-project.github.io/docling/','source':'src/backend/base/langflow/components/docling/export_docling_document.py#L11'},{'type':'ExtractKeyFromData','display_name':'Extract Key From Data','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/extract_key_from_data.py#L5'},{'type':'ExtractaKey','display_name':'Extract Key','inputs':{'data_input':[],'key':[]},'outputs':{'extracted_data':[]},'source':'src/backend/base/langflow/components/processing/extract_key.py#L6'},{'type':'FAISS','display_name':'FAISS','inputs':{'index_name':[],'persist_directory':[],'allow_dangerous_deserialization':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/faiss.py#L11'},{'type':'File','display_name':'File','inputs':{'use_multithreading':[],'concurrency_multithreading':[]},'outputs':{'message':[]},'documentation':'https://docs.langflow.org/components-data#file','source':'src/backend/base/langflow/components/data/file.py#L10'},{'type':'FilterData','display_name':'Filter Data','inputs':{'data':[],'filter_criteria':[]},'outputs':{'filtered_data':[]},'source':'src/backend/base/langflow/components/processing/filter_data.py#L6'},{'type':'FilterDataValues','display_name':'Filter Values','inputs':{'input_data':[],'filter_key':['Data'],'filter_value':['Data'],'operator':[]},'outputs':{'filtered_data':[]},'source':'src/backend/base/langflow/components/processing/filter_data_values.py#L8'},{'type':'FirecrawlCrawlApi','display_name':'Firecrawl Crawl API','inputs':{'api_key':[],'url':[],'timeout':[],'idempotency_key':[],'crawlerOptions':[],'scrapeOptions':[]},'outputs':{'data':[]},'documentation':'https://docs.firecrawl.dev/v1/api-reference/endpoint/crawl-post','source':'src/backend/base/langflow/components/firecrawl/firecrawl_crawl_api.py#L8'},{'type':'FirecrawlExtractApi','display_name':'Firecrawl Extract API','inputs':{'api_key':[],'urls':[],'prompt':[],'schema':[],'enable_web_search':[]},'outputs':{'data':[]},'documentation':'https://docs.firecrawl.dev/api-reference/endpoint/extract','source':'src/backend/base/langflow/components/firecrawl/firecrawl_extract_api.py#L14'},{'type':'FirecrawlMapApi','display_name':'Firecrawl Map API','inputs':{'api_key':[],'urls':[],'ignore_sitemap':[],'sitemap_only':[],'include_subdomains':[]},'outputs':{'data':[]},'documentation':'https://docs.firecrawl.dev/api-reference/endpoint/map','source':'src/backend/base/langflow/components/firecrawl/firecrawl_map_api.py#L11'},{'type':'FirecrawlScrapeApi','display_name':'Firecrawl Scrape API','inputs':{'api_key':[],'url':[],'timeout':[],'scrapeOptions':[],'extractorOptions':[]},'outputs':{'data':[]},'documentation':'https://docs.firecrawl.dev/api-reference/endpoint/scrape','source':'src/backend/base/langflow/components/firecrawl/firecrawl_scrape_api.py#L12'},{'type':'FlowTool','display_name':'Flow as Tool [Deprecated]','inputs':{'flow_name':[],'tool_name':[],'tool_description':[],'return_direct':[]},'outputs':{'api_build_tool':[]},'source':'src/backend/base/langflow/components/logic/flow_tool.py#L16'},{'type':'GetEnvVar','display_name':'Get Environment Variable','inputs':{'env_var_name':[]},'outputs':{'env_var_value':[]},'source':'src/backend/base/langflow/components/datastax/getenvvar.py#L9'},{'type':'GitExtractorComponent','display_name':'GitExtractor','inputs':{'repository_url':[]},'outputs':{'text_based_file_contents':[],'directory_structure':[],'repository_info':[],'statistics':[],'files_content':[]},'source':'src/backend/base/langflow/components/git/gitextractor.py#L16'},{'type':'GitLoaderComponent','display_name':'Git','inputs':{'repo_source':[],'repo_path':[],'clone_url':[],'branch':[],'file_filter':[],'content_filter':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/git/git.py#L15'},{'type':'GleanSearchAPIComponent','display_name':'Glean Search API','inputs':{'glean_api_url':[],'glean_access_token':[],'query':[],'page_size':[],'request_options':[]},'outputs':{'dataframe':[]},'documentation':'https://docs.langflow.org/Components/components-tools#glean-search-api','source':'src/backend/base/langflow/components/glean/glean_search_api.py#L101'},{'type':'GmailAPI','display_name':'Gmail','inputs':{'recipient_email':[],'subject':[],'body':[],'cc':[],'bcc':[],'is_html':[],'gmail_user_id':[],'max_results':[],'message_id':[],'thread_id':[],'query':[],'message_body':[],'label_name':[],'label_id':[],'label_ids':[],'label_list_visibility':[],'message_list_visibility':[],'page_token':[],'include_spam_trash':[],'format':[],'resource_name':[],'person_fields':[],'attachment_id':[],'file_name':[],'attachment':[]},'outputs':{},'documentation':'https://docs.composio.dev','source':'src/backend/base/langflow/components/composio/gmail_composio.py#L16'},{'type':'GmailLoaderComponent','display_name':'Gmail Loader','inputs':{'json_string':[],'label_ids':[],'max_results':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/google/gmail.py#L23'},{'type':'GoogleDriveComponent','display_name':'Google Drive Loader','inputs':{'json_string':[],'document_id':[]},'outputs':{'docs':[]},'source':'src/backend/base/langflow/components/google/google_drive.py#L16'},{'type':'GoogleDriveSearchComponent','display_name':'Google Drive Search','inputs':{'token_string':[],'query_item':[],'valid_operator':[],'search_term':[],'query_string':[]},'outputs':{'doc_urls':[],'doc_ids':[],'doc_titles':[],'Data':[]},'source':'src/backend/base/langflow/components/google/google_drive_search.py#L13'},{'type':'GoogleGenerativeAIModel','display_name':'Google Generative AI','inputs':{'max_output_tokens':[],'model_name':[],'api_key':[],'top_p':[],'temperature':[],'n':[],'top_k':[],'tool_model_enabled':[]},'outputs':{},'source':'src/backend/base/langflow/components/google/google_generative_ai.py#L22'},{'type':'GoogleOAuthToken','display_name':'Google OAuth Token','inputs':{'scopes':[],'oauth_credentials':[]},'outputs':{'output':[]},'documentation':'https://developers.google.com/identity/protocols/oauth2/web-server?hl=pt-br#python_1','source':'src/backend/base/langflow/components/google/google_oauth_token.py#L14'},{'type':'GoogleSearchAPI','display_name':'Google Search API [DEPRECATED]','inputs':{'google_api_key':[],'google_cse_id':[],'input_value':[],'k':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/google_search_api.py#L8'},{'type':'GoogleSearchAPICore','display_name':'Google Search API','inputs':{'google_api_key':[],'google_cse_id':[],'input_value':[],'k':[]},'outputs':{'results':[]},'source':'src/backend/base/langflow/components/google/google_search_api_core.py#L8'},{'type':'GoogleSerperAPI','display_name':'Google Serper API [DEPRECATED]','inputs':{'serper_api_key':[],'query':[],'k':[],'query_type':[],'query_params':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/google_serper_api.py#L29'},{'type':'GoogleSerperAPICore','display_name':'Serper Google Search API','inputs':{'serper_api_key':[],'input_value':[],'k':[]},'outputs':{'results':[]},'source':'src/backend/base/langflow/components/serper/google_serper_api_core.py#L9'},{'type':'Google Generative AI Embeddings','display_name':'Google Generative AI Embeddings','inputs':{'api_key':[],'model_name':[]},'outputs':{'embeddings':[]},'documentation':'https://python.langchain.com/v0.2/docs/integrations/text_embedding/google_generative_ai/','source':'src/backend/base/langflow/components/google/google_generative_ai_embeddings.py#L20'},{'type':'Graph RAG','display_name':'Graph RAG','inputs':{'embedding_model':['Embeddings'],'vector_store':['VectorStore'],'edge_definition':[],'strategy':[],'search_query':[],'graphrag_strategy_kwargs':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/graph_rag.py#L26'},{'type':'GroqModel','display_name':'Groq','inputs':{'api_key':[],'base_url':[],'max_tokens':[],'temperature':[],'n':[],'model_name':[],'tool_model_enabled':[]},'outputs':{},'source':'src/backend/base/langflow/components/groq/groq.py#L16'},{'type':'HCD','display_name':'Hyper-Converged Database','inputs':{'collection_name':[],'username':[],'password':[],'api_endpoint':[],'namespace':[],'ca_certificate':[],'metric':[],'batch_size':[],'bulk_insert_batch_concurrency':[],'bulk_insert_overwrite_concurrency':[],'bulk_delete_concurrency':[],'setup_mode':[],'pre_delete_collection':[],'metadata_indexing_include':[],'embedding':['Embeddings','dict'],'metadata_indexing_exclude':[],'collection_indexing_policy':[],'number_of_results':[],'search_type':[],'search_score_threshold':[],'search_filter':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/hcd.py#L16'},{'type':'HierarchicalCrewComponent','display_name':'Hierarchical Crew','inputs':{'agents':['Agent'],'tasks':['HierarchicalTask'],'manager_llm':['LanguageModel'],'manager_agent':['Agent']},'outputs':{},'documentation':'https://docs.crewai.com/how-to/Hierarchical/','source':'src/backend/base/langflow/components/crewai/hierarchical_crew.py#L5'},{'type':'HierarchicalTaskComponent','display_name':'Hierarchical Task','inputs':{'task_description':[],'expected_output':[],'tools':['Tool']},'outputs':{'task_output':[]},'source':'src/backend/base/langflow/components/crewai/hierarchical_task.py#L6'},{'type':'HomeAssistantControl','display_name':'Home Assistant Control','inputs':{'ha_token':[],'base_url':[],'default_action':[],'default_entity_id':[]},'outputs':{},'documentation':'https://developers.home-assistant.io/docs/api/rest/','source':'src/backend/base/langflow/components/homeassistant/home_assistant_control.py#L14'},{'type':'HtmlLinkExtractor','display_name':'HTML Link Extractor','inputs':{'kind':[],'drop_fragments':[],'data_input':['Document','Data']},'outputs':{},'documentation':'https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.extractors.html_link_extractor.HtmlLinkExtractor.html','source':'src/backend/base/langflow/components/langchain_utilities/html_link_extractor.py#L10'},{'type':'HuggingFaceModel','display_name':'Hugging Face','inputs':{'model_id':[],'custom_model':[],'max_new_tokens':[],'top_k':[],'top_p':[],'typical_p':[],'temperature':[],'repetition_penalty':[],'inference_endpoint':[],'task':[],'huggingfacehub_api_token':[],'model_kwargs':[],'retry_attempts':[]},'outputs':{},'source':'src/backend/base/langflow/components/huggingface/huggingface.py#L18'},{'type':'IBMwatsonxModel','display_name':'IBM watsonx.ai','inputs':{'url':[],'project_id':[],'api_key':[],'model_name':[],'max_tokens':[],'stop_sequence':[],'temperature':[],'top_p':[],'frequency_penalty':[],'presence_penalty':[],'seed':[],'logprobs':[],'top_logprobs':[],'logit_bias':[]},'outputs':{},'source':'src/backend/base/langflow/components/ibm/watsonx.py#L16'},{'type':'IDGenerator','display_name':'ID Generator','inputs':{'unique_id':[]},'outputs':{'id':[]},'source':'src/backend/base/langflow/components/helpers/id_generator.py#L12'},{'type':'JSONCleaner','display_name':'JSON Cleaner','inputs':{'json_str':[],'remove_control_chars':[],'normalize_unicode':[],'validate_json':[]},'outputs':{'output':[]},'source':'src/backend/base/langflow/components/processing/json_cleaner.py#L10'},{'type':'JSONDocumentBuilder','display_name':'JSON Document Builder','inputs':{'key':[],'document':[]},'outputs':{},'documentation':'https://docs.langflow.org/components/utilities#json-document-builder','source':'src/backend/base/langflow/components/deactivated/json_document_builder.py#L22'},{'type':'JSONtoData','display_name':'Load JSON','inputs':{'json_file':[],'json_path':[],'json_string':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/data/json_to_data.py#L11'},{'type':'JigsawStackAIScraper','display_name':'AI Scraper','inputs':{'api_key':[],'url':[],'html':[],'element_prompts':[],'root_element_selector':[]},'outputs':{'scrape_results':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/scrape','source':'src/backend/base/langflow/components/jigsawstack/ai_scrape.py#L8'},{'type':'JigsawStackAISearch','display_name':'AI Web Search','inputs':{'api_key':[],'query':[],'ai_overview':[],'safe_search':[],'spell_check':[]},'outputs':{'search_results':[],'content_text':[]},'documentation':'https://jigsawstack.com/docs/api-reference/web/ai-search','source':'src/backend/base/langflow/components/jigsawstack/ai_web_search.py#L7'},{'type':'JigsawStackFileRead','display_name':'File Read','inputs':{'api_key':[],'key':[]},'outputs':{'file_path':[]},'documentation':'https://jigsawstack.com/docs/api-reference/store/file/get','source':'src/backend/base/langflow/components/jigsawstack/file_read.py#L8'},{'type':'JigsawStackFileUpload','display_name':'File Upload','inputs':{'api_key':[],'file':[],'key':[],'overwrite':[],'temp_public_url':[]},'outputs':{'file_upload_result':[]},'documentation':'https://jigsawstack.com/docs/api-reference/store/file/add','source':'src/backend/base/langflow/components/jigsawstack/file_upload.py#L8'},{'type':'JigsawStackImageGeneration','display_name':'Image Generation','inputs':{'api_key':[],'prompt':[],'aspect_ratio':[],'url':[],'file_store_key':[],'width':[],'height':[],'steps':[],'output_format':[],'negative_prompt':[],'seed':[],'guidance':[]},'outputs':{'image_generation_results':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/image-generation','source':'src/backend/base/langflow/components/jigsawstack/image_generation.py#L6'},{'type':'JigsawStackNSFW','display_name':'NSFW Detection','inputs':{'api_key':[],'url':[]},'outputs':{'nsfw_result':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/nsfw','source':'src/backend/base/langflow/components/jigsawstack/nsfw.py#L6'},{'type':'JigsawStackObjectDetection','display_name':'Object Detection','inputs':{'api_key':[],'prompts':[],'url':[],'file_store_key':[],'annotated_image':[],'features':[],'return_type':[]},'outputs':{'object_detection_results':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/object-detection','source':'src/backend/base/langflow/components/jigsawstack/object_detection.py#L6'},{'type':'JigsawStackSentiment','display_name':'Sentiment Analysis','inputs':{'api_key':[],'text':[]},'outputs':{'sentiment_data':[],'sentiment_text':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/sentiment','source':'src/backend/base/langflow/components/jigsawstack/sentiment.py#L7'},{'type':'JigsawStackTextToSQL','display_name':'Text to SQL','inputs':{'api_key':[],'prompt':[],'sql_schema':[],'file_store_key':[]},'outputs':{'sql_query':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/text-to-sql','source':'src/backend/base/langflow/components/jigsawstack/text_to_sql.py#L6'},{'type':'JigsawStackTextTranslate','display_name':'Text Translate','inputs':{'api_key':[],'target_language':[],'text':[]},'outputs':{'translation_results':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/translate','source':'src/backend/base/langflow/components/jigsawstack/text_translate.py#L6'},{'type':'JigsawStackVOCR','display_name':'VOCR','inputs':{'api_key':[],'prompts':[],'url':[],'file_store_key':[],'page_range_start':[],'page_range_end':[]},'outputs':{'vocr_results':[]},'documentation':'https://jigsawstack.com/docs/api-reference/ai/vocr','source':'src/backend/base/langflow/components/jigsawstack/vocr.py#L6'},{'type':'JsonAgent','display_name':'JsonAgent','inputs':{'llm':['LanguageModel'],'path':[]},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/json_agent.py#L13'},{'type':'LLMCheckerChain','display_name':'LLMCheckerChain','inputs':{'input_value':[],'llm':['LanguageModel']},'outputs':{},'documentation':'https://python.langchain.com/docs/modules/chains/additional/llm_checker','source':'src/backend/base/langflow/components/langchain_utilities/llm_checker.py#L8'},{'type':'LLMMathChain','display_name':'LLMMathChain','inputs':{'input_value':[],'llm':['LanguageModel']},'outputs':{'text':[]},'documentation':'https://python.langchain.com/docs/modules/chains/additional/llm_math','source':'src/backend/base/langflow/components/langchain_utilities/llm_math.py#L9'},{'type':'LLMRouterComponent','display_name':'LLM Router','inputs':{'models':['LanguageModel'],'input_value':[],'judge_llm':['LanguageModel'],'optimization':[],'use_openrouter_specs':[],'timeout':[],'fallback_to_first':[]},'outputs':{'output':[],'selected_model_info':['Data'],'routing_decision':[]},'documentation':'https://docs.langflow.org/components-processing#llm-router','source':'src/backend/base/langflow/components/processing/llm_router.py#L17'},{'type':'LMStudioModel','display_name':'LM Studio','inputs':{'max_tokens':[],'model_kwargs':[],'model_name':[],'base_url':[],'api_key':[],'temperature':[],'seed':[]},'outputs':{},'source':'src/backend/base/langflow/components/lmstudio/lmstudiomodel.py#L14'},{'type':'LangChain Hub Prompt','display_name':'Prompt Hub','inputs':{'langchain_api_key':[],'langchain_hub_prompt':[]},'outputs':{'prompt':[]},'source':'src/backend/base/langflow/components/langchain_utilities/langchain_hub.py#L11'},{'type':'LangWatchEvaluator','display_name':'LangWatch Evaluator','inputs':{'evaluator_name':[],'api_key':[],'input':[],'output':[],'expected_output':[],'contexts':[],'timeout':[]},'outputs':{'evaluation_result':[]},'documentation':'https://docs.langwatch.ai/langevals/documentation/introduction','source':'src/backend/base/langflow/components/langwatch/langwatch.py#L25'},{'type':'LanguageModelComponent','display_name':'Language Model','inputs':{'provider':[],'model_name':[],'api_key':[],'input_value':['Message'],'system_message':['Message'],'stream':[],'temperature':[],'user_message':['Message']},'outputs':{'text_output':['Message']},'documentation':'https://docs.langflow.org/components-models','source':'src/backend/base/langflow/components/models/language_model.py#L18'},{'type':'LanguageRecursiveTextSplitter','display_name':'Language Recursive Text Splitter','inputs':{'chunk_size':[],'chunk_overlap':[],'data_input':['Document','Data'],'code_language':[]},'outputs':{},'documentation':'https://docs.langflow.org/components/text-splitters#languagerecursivetextsplitter','source':'src/backend/base/langflow/components/langchain_utilities/language_recursive.py#L9'},{'type':'ListFlows','display_name':'List Flows','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/list_flows.py#L5'},{'type':'ListHomeAssistantStates','display_name':'List Home Assistant States','inputs':{'ha_token':[],'base_url':[],'filter_domain':[]},'outputs':{},'documentation':'https://developers.home-assistant.io/docs/api/rest/','source':'src/backend/base/langflow/components/homeassistant/list_home_assistant_states.py#L14'},{'type':'Listen','display_name':'Listen','inputs':{'context_key':['Message']},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/logic/listen.py#L6'},{'type':'LocalDB','display_name':'Local DB','inputs':{'mode':[],'collection_name':[],'persist_directory':[],'existing_collections':[],'embedding':['Embeddings'],'allow_duplicates':[],'search_type':[],'ingest_data':['Data','DataFrame'],'search_query':[],'number_of_results':[],'limit':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/vectorstores/local_db.py#L17'},{'type':'LoopComponent','display_name':'Loop','inputs':{'data':['DataFrame']},'outputs':{'item':[],'done':[]},'documentation':'https://docs.langflow.org/components-logic#loop','source':'src/backend/base/langflow/components/logic/loop.py#L8'},{'type':'MCPSse','display_name':'MCP Tools (SSE) [DEPRECATED]','inputs':{'url':[]},'outputs':{'tools':[]},'documentation':'https://docs.langflow.org/components-custom-components','source':'src/backend/base/langflow/components/deactivated/mcp_sse.py#L17'},{'type':'MCPStdio','display_name':'MCP Tools (stdio) [DEPRECATED]','inputs':{'command':[]},'outputs':{'tools':[]},'documentation':'https://docs.langflow.org/components-custom-components','source':'src/backend/base/langflow/components/deactivated/mcp_stdio.py#L17'},{'type':'Maritalk','display_name':'MariTalk','inputs':{'max_tokens':[],'model_name':[],'api_key':[],'temperature':[]},'outputs':{},'source':'src/backend/base/langflow/components/maritalk/maritalk.py#L9'},{'type':'Memory','display_name':'Message History','inputs':{'mode':[],'message':[],'memory':['Memory'],'sender_type':[],'sender':[],'sender_name':[],'n_messages':[],'session_id':[],'order':[],'template':[]},'outputs':{'messages_text':[],'dataframe':[]},'documentation':'https://docs.langflow.org/components-helpers#message-history','source':'src/backend/base/langflow/components/helpers/memory.py#L16'},{'type':'MergeDataComponent','display_name':'Merge Data','inputs':{'data_inputs':[]},'outputs':{'merged_data':[]},'source':'src/backend/base/langflow/components/deactivated/merge_data.py#L8'},{'type':'Message','display_name':'Message','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/message.py#L6'},{'type':'MessagetoData','display_name':'Message to Data','inputs':{'message':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/processing/message_to_data.py#L9'},{'type':'MetalRetriever','display_name':'Metal Retriever','inputs':{'api_key':[],'client_id':[],'index_id':[],'params':[]},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/metal.py#L9'},{'type':'Milvus','display_name':'Milvus','inputs':{'collection_name':[],'collection_description':[],'uri':[],'password':[],'connection_args':[],'primary_field':[],'text_field':[],'vector_field':[],'consistency_level':[],'index_params':[],'search_params':[],'drop_old':[],'timeout':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/milvus.py#L16'},{'type':'MistalAIEmbeddings','display_name':'MistralAI Embeddings','inputs':{'model':[],'mistral_api_key':[],'max_concurrent_requests':[],'max_retries':[],'timeout':[],'endpoint':[]},'outputs':{'embeddings':[]},'source':'src/backend/base/langflow/components/mistral/mistral_embeddings.py#L9'},{'type':'MistralModel','display_name':'MistralAI','inputs':{'max_tokens':[],'model_name':[],'mistral_api_base':[],'api_key':[],'temperature':[],'max_retries':[],'timeout':[],'max_concurrent_requests':[],'top_p':[],'random_seed':[],'safe_mode':[]},'outputs':{},'source':'src/backend/base/langflow/components/mistral/mistral.py#L9'},{'type':'MongoDBAtlasVector','display_name':'MongoDB Atlas','inputs':{'mongodb_atlas_cluster_uri':[],'enable_mtls':[],'mongodb_atlas_client_cert':[],'db_name':[],'collection_name':[],'index_name':[],'insert_mode':[],'embedding':['Embeddings'],'number_of_results':[],'index_field':[],'filter_field':[],'number_dimensions':[],'similarity':[],'quantization':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/mongodb_atlas.py#L15'},{'type':'MultiQueryRetriever','display_name':'MultiQueryRetriever','inputs':{'llm':['LanguageModel'],'retriever':['BaseRetriever'],'prompt':[],'parser_key':[]},'outputs':{},'documentation':'https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/MultiQueryRetriever','source':'src/backend/base/langflow/components/deactivated/multi_query.py#L9'},{'type':'NVIDIAModelComponent','display_name':'NVIDIA','inputs':{'max_tokens':[],'model_name':[],'detailed_thinking':[],'tool_model_enabled':[],'base_url':[],'api_key':[],'temperature':[],'seed':[]},'outputs':{},'source':'src/backend/base/langflow/components/nvidia/nvidia.py#L14'},{'type':'NaturalLanguageTextSplitter','display_name':'Natural Language Text Splitter','inputs':{'chunk_size':[],'chunk_overlap':[],'data_input':['Document','Data'],'separator':[],'language':[]},'outputs':{},'documentation':'https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/split_by_token/#nltk','source':'src/backend/base/langflow/components/langchain_utilities/natural_language.py#L10'},{'type':'NewsSearch','display_name':'News Search','inputs':{'query':[],'hl':[],'gl':[],'ceid':[],'topic':[],'location':[],'timeout':[]},'outputs':{'articles':[]},'documentation':'https://docs.langflow.org/components-data#news-search','source':'src/backend/base/langflow/components/data/news_search.py#L12'},{'type':'NotDiamond','display_name':'Not Diamond Router','inputs':{'input_value':[],'system_message':[],'models':['LanguageModel'],'api_key':[],'preference_id':[],'tradeoff':[],'hash_content':[]},'outputs':{'output':[],'selected_model':[]},'documentation':'https://docs.notdiamond.ai/','source':'src/backend/base/langflow/components/notdiamond/notdiamond.py#L45'},{'type':'Notify','display_name':'Notify','inputs':{'context_key':[],'input_value':['Data','Message','DataFrame'],'append':[]},'outputs':{'result':[]},'source':'src/backend/base/langflow/components/logic/notify.py#L8'},{'type':'NotionDatabaseProperties','display_name':'List Database Properties ','inputs':{'database_id':[],'notion_secret':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/list-database-properties','source':'src/backend/base/langflow/components/Notion/list_database_properties.py#L12'},{'type':'NotionListPages','display_name':'List Pages ','inputs':{'notion_secret':[],'database_id':[],'query_json':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/list-pages','source':'src/backend/base/langflow/components/Notion/list_pages.py#L15'},{'type':'NotionPageContent','display_name':'Page Content Viewer ','inputs':{'page_id':[],'notion_secret':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/page-content-viewer','source':'src/backend/base/langflow/components/Notion/page_content_viewer.py#L12'},{'type':'NotionPageCreator','display_name':'Create Page ','inputs':{'database_id':[],'notion_secret':[],'properties_json':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/page-create','source':'src/backend/base/langflow/components/Notion/create_page.py#L14'},{'type':'NotionPageUpdate','display_name':'Update Page Property ','inputs':{'page_id':[],'properties':[],'notion_secret':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/page-update','source':'src/backend/base/langflow/components/Notion/update_page_property.py#L15'},{'type':'NotionSearch','display_name':'Search ','inputs':{'notion_secret':[],'query':[],'filter_value':[],'sort_direction':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/search','source':'src/backend/base/langflow/components/Notion/search.py#L13'},{'type':'NotionUserList','display_name':'List Users ','inputs':{'notion_secret':[]},'outputs':{},'documentation':'https://docs.langflow.org/integrations/notion/list-users','source':'src/backend/base/langflow/components/Notion/list_users.py#L11'},{'type':'NovitaModel','display_name':'Novita AI','inputs':{'max_tokens':[],'model_kwargs':[],'json_mode':[],'model_name':[],'api_key':[],'temperature':[],'seed':[],'output_parser':['OutputParser']},'outputs':{},'source':'src/backend/base/langflow/components/novita/novita.py#L21'},{'type':'NvidiaIngestComponent','display_name':'NVIDIA Retriever Extraction','inputs':{'base_url':[],'api_key':[],'extract_text':[],'extract_charts':[],'extract_tables':[],'extract_images':[],'extract_infographics':[],'text_depth':[],'split_text':[],'chunk_size':[],'chunk_overlap':[],'filter_images':[],'min_image_size':[],'min_aspect_ratio':[],'max_aspect_ratio':[],'dedup_images':[],'caption_images':[],'high_resolution':[]},'outputs':{},'documentation':'https://docs.nvidia.com/nemo/retriever/extraction/overview/','source':'src/backend/base/langflow/components/nvidia/nvidia_ingest.py#L10'},{'type':'NvidiaRerankComponent','display_name':'NVIDIA Rerank','inputs':{'api_key':[],'base_url':[],'model':[]},'outputs':{'reranked_documents':[]},'source':'src/backend/base/langflow/components/nvidia/nvidia_rerank.py#L11'},{'type':'OlivyaComponent','display_name':'Place Call','inputs':{'api_key':[],'from_number':[],'to_number':[],'first_message':[],'system_prompt':[],'conversation_history':[]},'outputs':{'output':[]},'documentation':'http://docs.langflow.org/components/olivya','source':'src/backend/base/langflow/components/olivya/olivya.py#L11'},{'type':'OllamaEmbeddings','display_name':'Ollama Embeddings','inputs':{'model_name':[],'base_url':[]},'outputs':{'embeddings':[]},'documentation':'https://python.langchain.com/docs/integrations/text_embedding/ollama','source':'src/backend/base/langflow/components/ollama/ollama_embeddings.py#L15'},{'type':'OllamaModel','display_name':'Ollama','inputs':{'base_url':[],'model_name':[],'temperature':[],'format':[],'metadata':[],'mirostat':[],'mirostat_eta':[],'mirostat_tau':[],'num_ctx':[],'num_gpu':[],'num_thread':[],'repeat_last_n':[],'repeat_penalty':[],'tfs_z':[],'timeout':[],'top_k':[],'top_p':[],'verbose':[],'tags':[],'stop_tokens':[],'system':[],'tool_model_enabled':[],'template':[]},'outputs':{},'source':'src/backend/base/langflow/components/ollama/ollama.py#L18'},{'type':'OpenAIModel','display_name':'OpenAI','inputs':{'max_tokens':[],'model_kwargs':[],'json_mode':[],'model_name':[],'openai_api_base':[],'api_key':[],'temperature':[],'seed':[],'max_retries':[],'timeout':[]},'outputs':{},'source':'src/backend/base/langflow/components/openai/openai_chat_model.py#L17'},{'type':'OpenAIToolsAgent','display_name':'OpenAI Tools Agent','inputs':{'llm':['LanguageModel','ToolEnabledLanguageModel'],'system_prompt':[],'user_prompt':[],'chat_history':[]},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/openai_tools.py#L13'},{'type':'OpenAPIAgent','display_name':'OpenAPI Agent','inputs':{'llm':['LanguageModel'],'path':[],'allow_dangerous_requests':[]},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/openapi.py#L14'},{'type':'OpenRouterComponent','display_name':'OpenRouter','inputs':{'api_key':[],'site_url':[],'app_name':[],'provider':[],'model_name':[],'temperature':[],'max_tokens':[]},'outputs':{},'source':'src/backend/base/langflow/components/openrouter/openrouter.py#L20'},{'type':'OpenSearch','display_name':'OpenSearch','inputs':{'opensearch_url':[],'index_name':[],'embedding':['Embeddings'],'search_type':[],'number_of_results':[],'search_score_threshold':[],'username':[],'password':[],'use_ssl':[],'verify_certs':[],'hybrid_search_query':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/opensearch.py#L22'},{'type':'OutputParser','display_name':'Output Parser','inputs':{'parser_type':[]},'outputs':{'format_instructions':[],'output_parser':[]},'source':'src/backend/base/langflow/components/helpers/output_parser.py#L9'},{'type':'ParseData','display_name':'Data to Message','inputs':{'data':[],'template':[],'sep':[]},'outputs':{'text':[],'data_list':[]},'source':'src/backend/base/langflow/components/processing/parse_data.py#L8'},{'type':'ParseDataFrame','display_name':'Parse DataFrame','inputs':{'df':[],'template':[],'sep':[]},'outputs':{'text':[]},'source':'src/backend/base/langflow/components/processing/parse_dataframe.py#L6'},{'type':'ParseJSONData','display_name':'Parse JSON','inputs':{'input_value':['Message','Data'],'query':[]},'outputs':{'filtered_data':[]},'source':'src/backend/base/langflow/components/processing/parse_json_data.py#L15'},{'type':'ParserComponent','display_name':'Parser','inputs':{'input_data':['DataFrame','Data'],'mode':[],'pattern':[],'sep':[]},'outputs':{'parsed_text':[]},'documentation':'https://docs.langflow.org/components-processing#parser','source':'src/backend/base/langflow/components/processing/parser.py#L10'},{'type':'Pass','display_name':'Pass','inputs':{'input_message':[],'ignored_message':[]},'outputs':{'output_message':[]},'source':'src/backend/base/langflow/components/logic/pass_message.py#L7'},{'type':'PerplexityModel','display_name':'Perplexity','inputs':{'model_name':[],'max_output_tokens':[],'api_key':[],'temperature':[],'top_p':[],'n':[],'top_k':[]},'outputs':{},'documentation':'https://python.langchain.com/v0.2/docs/integrations/chat/perplexity/','source':'src/backend/base/langflow/components/perplexity/perplexity.py#L10'},{'type':'Pinecone','display_name':'Pinecone','inputs':{'index_name':[],'namespace':[],'distance_strategy':[],'pinecone_api_key':[],'text_key':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/pinecone.py#L10'},{'type':'Prompt Template','display_name':'Prompt Template','inputs':{'template':['str'],'tool_placeholder':['Message'],'input':['str']},'outputs':{'prompt':['Message']},'documentation':'https://docs.langflow.org/components-prompts','source':'src/backend/base/langflow/components/processing/prompt.py#L9'},{'type':'PythonCodeStructuredTool','display_name':'Python Code Structured','inputs':{'tool_code':[],'tool_name':[],'tool_description':[],'return_direct':[],'tool_function':[],'global_variables':['Data'],'_classes':[],'_functions':[]},'outputs':{'result_tool':[]},'documentation':'https://python.langchain.com/docs/modules/tools/custom_tools/#structuredtool-dataclass','source':'src/backend/base/langflow/components/tools/python_code_structured_tool.py#L26'},{'type':'PythonFunction','display_name':'Python Function','inputs':{'function_code':[]},'outputs':{'function_output':[],'function_output_data':[],'function_output_str':[]},'source':'src/backend/base/langflow/components/prototypes/python_function.py#L13'},{'type':'PythonREPLComponent','display_name':'Python Interpreter','inputs':{'global_imports':[],'python_code':['Message']},'outputs':{'results':[]},'documentation':'https://docs.langflow.org/components-processing#python-interpreter','source':'src/backend/base/langflow/components/processing/python_repl_core.py#L10'},{'type':'PythonREPLTool','display_name':'Python REPL [DEPRECATED]','inputs':{'name':[],'description':[],'global_imports':[],'code':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/python_repl.py#L15'},{'type':'QdrantVectorStoreComponent','display_name':'Qdrant','inputs':{'collection_name':[],'host':[],'port':[],'grpc_port':[],'api_key':[],'prefix':[],'timeout':[],'path':[],'url':[],'distance_func':[],'content_payload_key':[],'metadata_payload_key':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/qdrant.py#L16'},{'type':'RSSReaderSimple','display_name':'RSS Reader','inputs':{'rss_url':[],'timeout':[]},'outputs':{'articles':[]},'documentation':'https://docs.langflow.org/components-data#rss-reader','source':'src/backend/base/langflow/components/data/rss.py#L11'},{'type':'RecursiveCharacterTextSplitter','display_name':'Recursive Character Text Splitter','inputs':{'chunk_size':[],'chunk_overlap':[],'data_input':['Document','Data'],'separators':[]},'outputs':{},'documentation':'https://docs.langflow.org/components-processing','source':'src/backend/base/langflow/components/langchain_utilities/recursive_character.py#L10'},{'type':'Redis','display_name':'Redis','inputs':{'redis_server_url':[],'redis_index_name':[],'code':[],'schema':[],'number_of_results':[],'embedding':['Embeddings']},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/redis.py#L12'},{'type':'RedisChatMemory','display_name':'Redis Chat Memory','inputs':{'host':[],'port':[],'database':[],'username':[],'password':[],'key_prefix':[],'session_id':[]},'outputs':{},'source':'src/backend/base/langflow/components/redis/redis.py#L10'},{'type':'RegexExtractorComponent','display_name':'Regex Extractor','inputs':{'input_text':[],'pattern':[]},'outputs':{'data':[],'text':[]},'source':'src/backend/base/langflow/components/processing/regex.py#L9'},{'type':'RetrievalQA','display_name':'Retrieval QA','inputs':{'input_value':[],'chain_type':[],'llm':['LanguageModel'],'retriever':['Retriever'],'memory':['BaseChatMemory'],'return_source_documents':[]},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/retrieval_qa.py#L10'},{'type':'RetrieverTool','display_name':'RetrieverTool','inputs':{'retriever':['Retriever'],'name':[],'description':[]},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/retriever.py#L8'},{'type':'RunFlow','display_name':'Run Flow','inputs':{},'outputs':{},'documentation':'https://docs.langflow.org/components-logic#run-flow','source':'src/backend/base/langflow/components/logic/run_flow.py#L10'},{'type':'RunnableExecutor','display_name':'Runnable Executor','inputs':{'input_value':[],'runnable':['Chain','AgentExecutor','Agent','Runnable'],'input_key':[],'output_key':[],'use_stream':[]},'outputs':{'text':[]},'source':'src/backend/base/langflow/components/langchain_utilities/runnable_executor.py#L9'},{'type':'SQLAgent','display_name':'SQLAgent','inputs':{'llm':['LanguageModel'],'database_uri':[],'extra_tools':['Tool']},'outputs':{'response':[],'agent':[]},'source':'src/backend/base/langflow/components/langchain_utilities/sql.py#L11'},{'type':'SQLDatabase','display_name':'SQLDatabase','inputs':{'uri':[]},'outputs':{'SQLDatabase':[]},'source':'src/backend/base/langflow/components/langchain_utilities/sql_database.py#L12'},{'type':'SQLGenerator','display_name':'Natural Language to SQL','inputs':{'input_value':[],'llm':['LanguageModel'],'db':['SQLDatabase'],'top_k':[],'prompt':[]},'outputs':{'text':[]},'source':'src/backend/base/langflow/components/langchain_utilities/sql_generator.py#L15'},{'type':'SambaNovaModel','display_name':'SambaNova','inputs':{'base_url':[],'model_name':[],'api_key':[],'max_tokens':[],'top_p':[],'temperature':[]},'outputs':{},'documentation':'https://cloud.sambanova.ai/','source':'src/backend/base/langflow/components/sambanova/sambanova.py#L11'},{'type':'SaveToFile','display_name':'Save File','inputs':{'input':['Data','DataFrame','Message'],'file_name':[],'file_format':[]},'outputs':{'result':[]},'documentation':'https://docs.langflow.org/components-processing#save-file','source':'src/backend/base/langflow/components/processing/save_file.py#L20'},{'type':'ScrapeGraphMarkdownifyApi','display_name':'ScrapeGraph Markdownify API','inputs':{'api_key':[],'url':[]},'outputs':{'data':[]},'documentation':'https://docs.scrapegraphai.com/services/markdownify','source':'src/backend/base/langflow/components/scrapegraph/scrapegraph_markdownify_api.py#L10'},{'type':'ScrapeGraphSearchApi','display_name':'ScrapeGraph Search API','inputs':{'api_key':[],'user_prompt':[]},'outputs':{'data':[]},'documentation':'https://docs.scrapegraphai.com/services/searchscraper','source':'src/backend/base/langflow/components/scrapegraph/scrapegraph_search_api.py#L10'},{'type':'ScrapeGraphSmartScraperApi','display_name':'ScrapeGraph Smart Scraper API','inputs':{'api_key':[],'url':[],'prompt':[]},'outputs':{'data':[]},'documentation':'https://docs.scrapegraphai.com/services/smartscraper','source':'src/backend/base/langflow/components/scrapegraph/scrapegraph_smart_scraper_api.py#L10'},{'type':'SearXNGTool','display_name':'SearXNG Search','inputs':{'url':[],'max_results':[],'categories':[],'language':[]},'outputs':{'result_tool':[]},'source':'src/backend/base/langflow/components/tools/searxng.py#L17'},{'type':'SearchAPI','display_name':'Search API [DEPRECATED]','inputs':{'engine':[],'api_key':[],'input_value':[],'search_params':[],'max_results':[],'max_snippet_length':[]},'outputs':{},'documentation':'https://www.searchapi.io/docs/google','source':'src/backend/base/langflow/components/tools/search_api.py#L13'},{'type':'SearchComponent','display_name':'SearchApi','inputs':{'engine':[],'api_key':[],'input_value':[],'search_params':[],'max_results':[],'max_snippet_length':[]},'outputs':{'dataframe':[]},'documentation':'https://www.searchapi.io/docs/google','source':'src/backend/base/langflow/components/searchapi/search.py#L12'},{'type':'SelectData','display_name':'Select Data','inputs':{'data_list':[],'data_index':[]},'outputs':{'selected_data':[]},'source':'src/backend/base/langflow/components/processing/select_data.py#L8'},{'type':'SelectivePassThrough','display_name':'Selective Pass Through','inputs':{'input_value':[],'comparison_value':[],'operator':[],'value_to_pass':[],'case_sensitive':[]},'outputs':{'passed_output':[]},'source':'src/backend/base/langflow/components/deactivated/selective_passthrough.py#L6'},{'type':'SelfQueryRetriever','display_name':'Self Query Retriever','inputs':{'query':['Message'],'vectorstore':['VectorStore'],'attribute_infos':['Data'],'document_content_description':[],'llm':['LanguageModel']},'outputs':{'documents':[]},'source':'src/backend/base/langflow/components/langchain_utilities/self_query.py#L11'},{'type':'SemanticTextSplitter','display_name':'Semantic Text Splitter','inputs':{'data_inputs':['Data'],'embeddings':['Embeddings'],'breakpoint_threshold_type':[],'breakpoint_threshold_amount':[],'number_of_chunks':[],'sentence_split_regex':[],'buffer_size':[]},'outputs':{'chunks':[]},'documentation':'https://python.langchain.com/docs/how_to/semantic-chunker/','source':'src/backend/base/langflow/components/langchain_utilities/language_semantic.py#L16'},{'type':'SequentialCrewComponent','display_name':'Sequential Crew','inputs':{'tasks':['SequentialTask']},'outputs':{},'documentation':'https://docs.crewai.com/how-to/Sequential/','source':'src/backend/base/langflow/components/crewai/sequential_crew.py#L6'},{'type':'SequentialTaskAgentComponent','display_name':'Sequential Task Agent','inputs':{'role':[],'goal':[],'backstory':[],'tools':['Tool'],'llm':['LanguageModel'],'memory':[],'verbose':[],'allow_delegation':[],'allow_code_execution':[],'agent_kwargs':[],'task_description':[],'expected_output':[],'async_execution':[],'previous_task':['SequentialTask']},'outputs':{'task_output':[]},'documentation':'https://docs.crewai.com/how-to/LLM-Connections/','source':'src/backend/base/langflow/components/crewai/sequential_task_agent.py#L6'},{'type':'SequentialTaskComponent','display_name':'Sequential Task','inputs':{'task_description':[],'expected_output':[],'tools':['Tool'],'agent':['Agent'],'task':['SequentialTask'],'async_execution':[]},'outputs':{'task_output':[]},'source':'src/backend/base/langflow/components/crewai/sequential_task.py#L6'},{'type':'Serp','display_name':'Serp Search API','inputs':{'serpapi_api_key':[],'input_value':[],'search_params':[],'max_results':[],'max_snippet_length':[]},'outputs':{'data':[],'text':[]},'source':'src/backend/base/langflow/components/serpapi/serp.py#L32'},{'type':'SerpAPI','display_name':'Serp Search API [DEPRECATED]','inputs':{'serpapi_api_key':[],'input_value':[],'search_params':[],'max_results':[],'max_snippet_length':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/serp_api.py#L32'},{'type':'ShouldRunNext','display_name':'Should Run Next','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/should_run_next.py#L8'},{'type':'Smart Function','display_name':'Smart Function','inputs':{'data':[],'llm':['LanguageModel'],'filter_instruction':[],'sample_size':[],'max_size':[]},'outputs':{'filtered_data':[]},'documentation':'https://docs.langflow.org/components-processing#smart-function','source':'src/backend/base/langflow/components/processing/lambda_filter.py#L16'},{'type':'SpiderTool','display_name':'Spider Web Crawler & Scraper','inputs':{'spider_api_key':[],'url':[],'mode':[],'limit':[],'depth':[],'blacklist':[],'whitelist':[],'readability':[],'request_timeout':[],'metadata':[],'params':[]},'outputs':{'content':[]},'documentation':'https://spider.cloud/docs/api','source':'src/backend/base/langflow/components/langchain_utilities/spider.py#L17'},{'type':'SplitText','display_name':'Split Text','inputs':{'data_inputs':['Data'],'chunk_overlap':[],'chunk_size':[],'separator':[]},'outputs':{'chunks':[]},'source':'src/backend/base/langflow/components/deactivated/split_text.py#L9'},{'type':'SplitVideo','display_name':'Split Video','inputs':{'videodata':['Data'],'clip_duration':[],'last_clip_handling':[],'include_original':[]},'outputs':{'clips':[]},'documentation':'https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md','source':'src/backend/base/langflow/components/twelvelabs/split_video.py#L14'},{'type':'StoreMessage','display_name':'Message Store','inputs':{'message':[],'memory':['Memory'],'sender':[],'sender_name':[],'session_id':[]},'outputs':{'stored_messages':[]},'source':'src/backend/base/langflow/components/helpers/store_message.py#L12'},{'type':'StructuredOutput','display_name':'Structured Output','inputs':{'llm':['LanguageModel'],'input_value':[],'system_prompt':[],'schema_name':[],'output_schema':[]},'outputs':{'structured_output':[],'dataframe_output':[]},'documentation':'https://docs.langflow.org/components-processing#structured-output','source':'src/backend/base/langflow/components/processing/structured_output.py#L19'},{'type':'SubFlow','display_name':'Sub Flow','inputs':{},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/sub_flow.py#L18'},{'type':'SupabaseVectorStore','display_name':'Supabase','inputs':{'supabase_url':[],'supabase_service_key':[],'table_name':[],'query_name':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/supabase.py#L10'},{'type':'TavilyAISearch','display_name':'Tavily Search API','inputs':{'api_key':[],'query':[],'search_depth':[],'chunks_per_source':[],'topic':[],'days':[],'max_results':[],'include_answer':[],'time_range':[],'include_images':[],'include_domains':[],'exclude_domains':[],'include_raw_content':[]},'outputs':{},'documentation':'https://docs.tavily.com/','source':'src/backend/base/langflow/components/tools/tavily_search_tool.py#L73'},{'type':'TavilyExtractComponent','display_name':'Tavily Extract API','inputs':{'api_key':[],'urls':[],'extract_depth':[],'include_images':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/tavily/tavily_extract.py#L10'},{'type':'TavilySearchComponent','display_name':'Tavily Search API','inputs':{'api_key':[],'query':[],'search_depth':[],'chunks_per_source':[],'topic':[],'days':[],'max_results':[],'include_answer':[],'time_range':[],'include_images':[],'include_domains':[],'exclude_domains':[],'include_raw_content':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/tavily/tavily_search.py#L11'},{'type':'TextEmbedderComponent','display_name':'Text Embedder','inputs':{'embedding_model':['Embeddings'],'message':[]},'outputs':{'embeddings':[]},'source':'src/backend/base/langflow/components/embeddings/text_embedder.py#L13'},{'type':'TextInput','display_name':'Text Input','inputs':{'input_value':['str']},'outputs':{'text':['str']},'documentation':'https://docs.langflow.org/components-io#text-input','source':'src/backend/base/langflow/components/input_output/text.py#L6'},{'type':'TextOutput','display_name':'Text Output','inputs':{'input_value':[]},'outputs':{'text':[]},'documentation':'https://docs.langflow.org/components-io#text-output','source':'src/backend/base/langflow/components/input_output/text_output.py#L6'},{'type':'ToolCallingAgent','display_name':'Tool Calling Agent','inputs':{'llm':['LanguageModel'],'system_prompt':[],'chat_history':[]},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/tool_calling.py#L13'},{'type':'TwelveLabsPegasus','display_name':'TwelveLabs Pegasus','inputs':{'videodata':[],'api_key':[],'video_id':[],'index_name':[],'index_id':[],'model_name':[],'message':[],'temperature':[]},'outputs':{'response':[],'processed_video_id':[]},'documentation':'https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md','source':'src/backend/base/langflow/components/twelvelabs/twelvelabs_pegasus.py#L37'},{'type':'TwelveLabsPegasusIndexVideo','display_name':'TwelveLabs Pegasus Index Video','inputs':{'videodata':[],'api_key':[],'model_name':[],'index_name':[],'index_id':[]},'outputs':{'indexed_data':[]},'documentation':'https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md','source':'src/backend/base/langflow/components/twelvelabs/pegasus_index.py#L31'},{'type':'TypeConverterComponent','display_name':'Type Convert','inputs':{'input_data':['Message','Data','DataFrame'],'output_type':[]},'outputs':{'message_output':[]},'documentation':'https://docs.langflow.org/components-processing#type-convert','source':'src/backend/base/langflow/components/processing/converter.py#L50'},{'type':'URLComponent','display_name':'URL','inputs':{'urls':[],'max_depth':[],'prevent_outside':[],'use_async':[],'format':[],'timeout':[],'headers':['DataFrame'],'filter_text_html':[],'continue_on_failure':[],'check_response_status':[],'autoset_encoding':[]},'outputs':{'page_results':[],'raw_results':[]},'documentation':'https://docs.langflow.org/components-data#url','source':'src/backend/base/langflow/components/data/url.py#L26'},{'type':'Unstructured','display_name':'Unstructured API','inputs':{'api_key':[],'api_url':[],'chunking_strategy':[],'unstructured_args':[]},'outputs':{},'documentation':'https://python.langchain.com/api_reference/unstructured/document_loaders/langchain_unstructured.document_loaders.UnstructuredLoader.html','source':'src/backend/base/langflow/components/unstructured/unstructured.py#L8'},{'type':'UpdateData','display_name':'Update Data','inputs':{'old_data':[],'number_of_fields':[],'text_key':[],'text_key_validator':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/processing/update_data.py#L17'},{'type':'Upstash','display_name':'Upstash','inputs':{'index_url':[],'index_token':[],'text_key':[],'namespace':[],'metadata_filter':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/upstash.py#L15'},{'type':'Vectara','display_name':'Vectara','inputs':{'vectara_customer_id':[],'vectara_corpus_id':[],'vectara_api_key':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/vectara.py#L14'},{'type':'VectaraRAG','display_name':'Vectara RAG','inputs':{'vectara_customer_id':[],'vectara_corpus_id':[],'vectara_api_key':[],'search_query':[],'lexical_interpolation':[],'filter':[],'reranker':[],'reranker_k':[],'diversity_bias':[],'max_results':[],'response_lang':[],'prompt':[]},'outputs':{'answer':[]},'documentation':'https://docs.vectara.com/docs','source':'src/backend/base/langflow/components/vectorstores/vectara_rag.py#L7'},{'type':'VectaraSelfQueryRetriver','display_name':'Vectara Self Query Retriever','inputs':{'vectorstore':[],'llm':[],'document_content_description':[],'metadata_field_info':[]},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/vectara_self_query.py#L12'},{'type':'VectorStoreInfo','display_name':'VectorStoreInfo','inputs':{'vectorstore_name':[],'vectorstore_description':[],'input_vectorstore':['VectorStore']},'outputs':{'info':[]},'source':'src/backend/base/langflow/components/langchain_utilities/vector_store_info.py#L8'},{'type':'VectorStoreRetriever','display_name':'VectorStore Retriever','inputs':{'vectorstore':['VectorStore']},'outputs':{},'source':'src/backend/base/langflow/components/deactivated/vector_store.py#L8'},{'type':'VectorStoreRouterAgent','display_name':'VectorStoreRouterAgent','inputs':{'llm':['LanguageModel'],'vectorstores':['VectorStoreInfo']},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/vector_store_router.py#L8'},{'type':'VertexAIEmbeddings','display_name':'Vertex AI Embeddings','inputs':{'credentials':[],'location':[],'project':[],'max_output_tokens':[],'max_retries':[],'model_name':[],'n':[],'request_parallelism':[],'stop_sequences':[],'streaming':[],'temperature':[],'top_k':[],'top_p':[]},'outputs':{'embeddings':[]},'source':'src/backend/base/langflow/components/vertexai/vertexai_embeddings.py#L6'},{'type':'VertexAiModel','display_name':'Vertex AI','inputs':{'credentials':[],'model_name':[],'project':[],'location':[],'max_output_tokens':[],'max_retries':[],'temperature':[],'top_k':[],'top_p':[],'verbose':[]},'outputs':{},'source':'src/backend/base/langflow/components/vertexai/vertexai.py#L9'},{'type':'VideoFile','display_name':'Video File','inputs':{'file_path':[]},'outputs':{},'documentation':'https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md','source':'src/backend/base/langflow/components/twelvelabs/video_file.py#L8'},{'type':'Weaviate','display_name':'Weaviate','inputs':{'url':[],'api_key':[],'index_name':[],'text_key':[],'embedding':['Embeddings'],'number_of_results':[],'search_by_text':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/weaviate.py#L10'},{'type':'WebSearchNoAPI','display_name':'Web Search','inputs':{'query':[],'timeout':[]},'outputs':{'results':[]},'documentation':'https://docs.langflow.org/components-data#web-search','source':'src/backend/base/langflow/components/data/web_search.py#L14'},{'type':'Webhook','display_name':'Webhook','inputs':{'data':[],'curl':[],'endpoint':[]},'outputs':{'output_data':[]},'documentation':'https://docs.langflow.org/components-data#webhook','source':'src/backend/base/langflow/components/data/webhook.py#L8'},{'type':'WikidataAPI','display_name':'Wikidata API [Deprecated]','inputs':{'query':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/wikidata_api.py#L55'},{'type':'WikidataComponent','display_name':'Wikidata','inputs':{'query':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/wikipedia/wikidata.py#L12'},{'type':'WikipediaAPI','display_name':'Wikipedia API [Deprecated]','inputs':{'input_value':[],'lang':[],'k':[],'load_all_available_meta':[],'doc_content_chars_max':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/wikipedia_api.py#L12'},{'type':'WikipediaComponent','display_name':'Wikipedia','inputs':{'input_value':[],'lang':[],'k':[],'load_all_available_meta':[],'doc_content_chars_max':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/wikipedia/wikipedia.py#L10'},{'type':'WolframAlphaAPI','display_name':'WolframAlpha API','inputs':{'input_value':[],'app_id':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/wolframalpha/wolfram_alpha_api.py#L11'},{'type':'XMLAgent','display_name':'XML Agent','inputs':{'llm':['LanguageModel'],'chat_history':[],'system_prompt':[],'user_prompt':[]},'outputs':{},'source':'src/backend/base/langflow/components/langchain_utilities/xml_agent.py#L13'},{'type':'YahooFinanceTool','display_name':'Yahoo! Finance [DEPRECATED]','inputs':{'symbol':[],'method':[],'num_news':[]},'outputs':{},'source':'src/backend/base/langflow/components/tools/yahoo_finance.py#L51'},{'type':'YfinanceComponent','display_name':'Yahoo! Finance','inputs':{'symbol':[],'method':[],'num_news':[]},'outputs':{'dataframe':[]},'source':'src/backend/base/langflow/components/yahoosearch/yahoo.py#L51'},{'type':'YouTubeChannelComponent','display_name':'YouTube Channel','inputs':{'channel_url':[],'api_key':[],'include_statistics':[],'include_branding':[],'include_playlists':[]},'outputs':{'channel_df':[]},'source':'src/backend/base/langflow/components/youtube/channel.py#L14'},{'type':'YouTubeCommentsComponent','display_name':'YouTube Comments','inputs':{'video_url':[],'api_key':[],'max_results':[],'sort_by':[],'include_replies':[],'include_metrics':[]},'outputs':{'comments':[]},'source':'src/backend/base/langflow/components/youtube/comments.py#L13'},{'type':'YouTubePlaylistComponent','display_name':'YouTube Playlist','inputs':{'playlist_url':[]},'outputs':{'video_urls':[]},'source':'src/backend/base/langflow/components/youtube/playlist.py#L10'},{'type':'YouTubeSearchComponent','display_name':'YouTube Search','inputs':{'query':[],'api_key':[],'max_results':[],'order':[],'include_metadata':[]},'outputs':{'results':[]},'source':'src/backend/base/langflow/components/youtube/search.py#L13'},{'type':'YouTubeTranscripts','display_name':'YouTube Transcripts','inputs':{'url':[],'chunk_size_seconds':[],'translation':[]},'outputs':{'dataframe':[],'message':[],'data_output':[]},'source':'src/backend/base/langflow/components/youtube/youtube_transcripts.py#L14'},{'type':'YouTubeTrendingComponent','display_name':'YouTube Trending','inputs':{'api_key':[],'region':[],'category':[],'max_results':[],'include_statistics':[],'include_content_details':[],'include_thumbnails':[]},'outputs':{'trending_videos':[]},'source':'src/backend/base/langflow/components/youtube/trending.py#L17'},{'type':'YouTubeVideoDetailsComponent','display_name':'YouTube Video Details','inputs':{'video_url':[],'api_key':[],'include_statistics':[],'include_content_details':[],'include_tags':[],'include_thumbnails':[]},'outputs':{'video_data':[]},'source':'src/backend/base/langflow/components/youtube/video_details.py#L14'},{'type':'ZepChatMemory','display_name':'Zep Chat Memory','inputs':{'url':[],'api_key':[],'api_base_path':[],'session_id':[]},'outputs':{},'source':'src/backend/base/langflow/components/zep/zep.py#L6'},{'type':'mem0_chat_memory','display_name':'Mem0 Chat Memory','inputs':{'mem0_config':['Data'],'ingest_message':[],'existing_memory':['Memory'],'user_id':[],'search_query':[],'mem0_api_key':[],'metadata':[],'openai_api_key':[]},'outputs':{'memory':[],'search_results':[]},'source':'src/backend/base/langflow/components/mem0/mem0_chat_memory.py#L18'},{'type':'needle','display_name':'Needle Retriever','inputs':{'needle_api_key':[],'collection_id':[],'query':[],'top_k':[]},'outputs':{'result':[]},'documentation':'https://docs.needle-ai.com','source':'src/backend/base/langflow/components/needle/needle.py#L9'},{'type':'pgvector','display_name':'PGVector','inputs':{'pg_server_url':[],'collection_name':[],'embedding':['Embeddings'],'number_of_results':[]},'outputs':{},'source':'src/backend/base/langflow/components/vectorstores/pgvector.py#L10'},{'type':'s3bucketuploader','display_name':'S3 Bucket Uploader','inputs':{'aws_access_key_id':[],'aws_secret_access_key':[],'bucket_name':[],'strategy':[],'data_inputs':['Data'],'s3_prefix':[],'strip_path':[]},'outputs':{'data':[]},'source':'src/backend/base/langflow/components/amazon/s3_bucket_uploader.py#L15'},{'type':'xAIModel','display_name':'xAI','inputs':{'max_tokens':[],'model_kwargs':[],'json_mode':[],'model_name':[],'base_url':[],'api_key':[],'temperature':[],'seed':[]},'outputs':{},'source':'src/backend/base/langflow/components/xai/xai.py#L22'}]}",
            "",
            "Execute this step now:"
        ])

        
        final_prompt = "\n".join(parts)
        
        # CLI logging for dynamic step prompt
        print(f"\n{'='*80}")
        print(f"🧠 DYNAMIC STEP PROMPT")
        print(f"{'='*80}")
        print(final_prompt)
        print(f"{'='*80}")
        print(f"📊 DYNAMIC STEP PROMPT STATISTICS:")
        print(f"   Total length: {len(final_prompt)} characters")
        print(f"   Total lines: {final_prompt.count(chr(10)) + 1}")
        print(f"   Flow data included: {'Yes' if flow_data else 'No'}")
        print(f"   Snippets included: {len(retrieved_snippets) if retrieved_snippets else 0}")
        print(f"{'='*80}\n")
        
        return final_prompt
    
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
    
    async def execute_single_step(self, step_name: str, user_prompt: str, flow_data: dict | None, retrieved_snippets: list[dict[str, Any]], previous_steps: dict[str, Any] | None = None, available_templates: list[str] | None = None) -> dict[str, Any]:
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
            retrieved_snippets, previous_steps or {}, step_config, available_templates
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