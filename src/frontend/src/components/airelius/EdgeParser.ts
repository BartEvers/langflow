// EdgeParser.ts - Handles parsing edge patterns and generating Langflow edge JSON

export interface EdgeData {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
  data: {
    targetHandle: {
      fieldName: string;
      id: string;
      inputTypes: string[];
      type: string;
    };
    sourceHandle: {
    dataType: string;
      id: string;
    name: string;
    output_types: string[];
  };
  };
  id: string;
}



export class EdgeParser {
  private static instance: EdgeParser;
  private componentRegistry: any = null;
  private instancesMap: Map<string, { id: string; type: string }> = new Map();
  
  private constructor() {}
  
  public static getInstance(): EdgeParser {
    if (!EdgeParser.instance) {
      EdgeParser.instance = new EdgeParser();
    }
    return EdgeParser.instance;
  }

  /**
   * Set the component registry for type resolution
   */
  public setComponentRegistry(registry: any): void {
    this.componentRegistry = registry;
  }

  /**
   * Set the instances map for current canvas node types
   */
  public setInstancesMap(instances: Map<string, { id: string; type: string }>): void {
    this.instancesMap = instances;
  }

  /**
   * Parse edge definitions from component tag text
   * Format: "LanguageModelComponent-lRD6W.out:text_output -> TextInput-xyz123.in:input_value, TextInput-xyz123.out:text -> Prompt-3Um6W.in:template"
   */
  public parseEdgeDefinitions(edgeText: string, componentType: string, componentId: string): EdgeData[] {
    console.log('🔍 NEW-EDGE-PARSER: ===== STARTING EDGE PARSING =====');
    console.log('🔍 NEW-EDGE-PARSER: Input edgeText:', edgeText);
    console.log('🔍 NEW-EDGE-PARSER: Input edgeText length:', edgeText.length);
    console.log('🔍 NEW-EDGE-PARSER: Input edgeText chars:', Array.from(edgeText).map((char, i) => `${i}:${char}`).join(' '));
    console.log('🔍 NEW-EDGE-PARSER: Input componentType:', componentType);
    console.log('🔍 NEW-EDGE-PARSER: Input componentId:', componentId);
    
    if (!edgeText || !edgeText.trim()) {
      console.log('🔍 NEW-EDGE-PARSER: No edge text provided, returning empty array');
      return [];
    }

    const edges: EdgeData[] = [];
    
    // Split by commas first, then parse each edge individually
    const edgeParts = edgeText.split(',').map(part => part.trim()).filter(part => part.includes('->'));
    
    console.log('🔍 NEW-EDGE-PARSER: Split edge parts:', edgeParts);
    
    for (let i = 0; i < edgeParts.length; i++) {
      const edgePart = edgeParts[i];
      const isFirstEdge = i === 0;
      
      // Parse individual edge part
      const edgeMatch = edgePart.match(/^([^.]+)\.out:([^->]+)\s*->\s*([^.]+)\.in:([^>\s]+)/);
      
      if (edgeMatch) {
        console.log('🔍 NEW-EDGE-PARSER: ===== FOUND EDGE MATCH =====');
        console.log('🔍 NEW-EDGE-PARSER: Edge part:', edgePart);
        console.log('🔍 NEW-EDGE-PARSER: Match groups:', edgeMatch);
        console.log('🔍 NEW-EDGE-PARSER: Is first edge:', isFirstEdge);
        
        const [, sourceNode, sourceHandle, targetNode, targetHandle] = edgeMatch;
        
        // Clean up the captured values
        const source = sourceNode.trim();
        const target = targetNode.trim();
        const sourceHandleName = sourceHandle.trim();
        const targetHandleName = targetHandle.trim();
        
        // Clean the node IDs by removing any edge content (everything after and including the first colon)
        const cleanSource = source.split(':')[0].trim();
        const cleanTarget = target.split(':')[0].trim();
        
        console.log('🔍 NEW-EDGE-PARSER: ===== PARSED EDGE COMPONENTS =====');
        console.log('🔍 NEW-EDGE-PARSER: Raw source:', source);
        console.log('🔍 NEW-EDGE-PARSER: Clean source:', cleanSource);
        console.log('🔍 NEW-EDGE-PARSER: sourceHandleName:', sourceHandleName);
        console.log('🔍 NEW-EDGE-PARSER: Raw target:', target);
        console.log('🔍 NEW-EDGE-PARSER: Clean target:', cleanTarget);
        console.log('🔍 NEW-EDGE-PARSER: targetHandleName:', targetHandleName);
        
        // Resolve node types from instances map or use prefix heuristic
        const srcType = this.instancesMap.get(cleanSource)?.type || this.getPrefixHeuristic(cleanSource);
        const dstType = this.instancesMap.get(cleanTarget)?.type || this.getPrefixHeuristic(cleanTarget);
        
        // Initialize dataType with srcType as fallback
        let dataType = srcType;
        
        console.log('🔍 NEW-EDGE-PARSER: Resolved types - srcType:', srcType, 'dstType:', dstType);
        console.log('🔍 NEW-EDGE-PARSER: Looking for cleanSource:', cleanSource, 'cleanTarget:', cleanTarget);
        

        
        // Look up types in the registry
        let outTypes: string[] = [];
        let inTypes: string[] = [];
        let targetType = 'str';  // Default fallback for type field
        
        if (this.componentRegistry?.components) {
          console.log('🔍 NEW-EDGE-PARSER: Registry has', this.componentRegistry.components.length, 'components');
          console.log('🔍 NEW-EDGE-PARSER: Searching for srcType:', srcType, 'dstType:', dstType);
          
          // Debug: Show what component types are actually in the registry
          const registryTypes = this.componentRegistry.components.map((c: any) => c.type);
          console.log('🔍 NEW-EDGE-PARSER: Available registry types:', registryTypes.slice(0, 10), '...');
          
          // Try exact match first
          let srcComponent = this.componentRegistry.components.find((c: any) => c.type === srcType);
          let dstComponent = this.componentRegistry.components.find((c: any) => c.type === dstType);
          
          // If no exact match, try hardcoded fallbacks for known components
          if (!srcComponent) {
            console.log('🔍 NEW-EDGE-PARSER: No exact match for srcType:', srcType, '- trying hardcoded fallbacks...');
            
            // Hardcoded fallbacks for known components
            if (srcType === 'Prompt') {
              srcComponent = this.componentRegistry.components.find((c: any) => c.type === 'Prompt Template');
              console.log('🔍 NEW-EDGE-PARSER: Looking for Prompt Template in registry...');
            } else if (srcType === 'TextInput') {
              srcComponent = this.componentRegistry.components.find((c: any) => c.type === 'Text Input');
              console.log('🔍 NEW-EDGE-PARSER: Looking for Text Input in registry...');
            } else if (srcType === 'Language Model') {
              srcComponent = this.componentRegistry.components.find((c: any) => c.type === 'Language Model');
              console.log('🔍 NEW-EDGE-PARSER: Looking for Language Model in registry...');
            }
            
            if (srcComponent) {
              console.log('🔍 NEW-EDGE-PARSER: Found srcComponent via hardcoded fallback:', srcComponent.type);
              console.log('🔍 NEW-EDGE-PARSER: srcComponent data:', JSON.stringify(srcComponent, null, 2));
            } else {
              console.log('🔍 NEW-EDGE-PARSER: Still no srcComponent found after hardcoded fallbacks');
            }
          }
          
          if (!dstComponent) {
            console.log('🔍 NEW-EDGE-PARSER: No exact match for dstType:', dstType, '- trying hardcoded fallbacks...');
            
            // Hardcoded fallbacks for known components
            if (dstType === 'Prompt') {
              dstComponent = this.componentRegistry.components.find((c: any) => c.type === 'Prompt Template');
            } else if (dstType === 'TextInput') {
              dstComponent = this.componentRegistry.components.find((c: any) => c.type === 'Text Input');
            } else if (dstType === 'Language Model') {
              dstComponent = this.componentRegistry.components.find((c: any) => c.type === 'Language Model');
            }
            
            if (dstComponent) {
              console.log('🔍 NEW-EDGE-PARSER: Found dstComponent via hardcoded fallback:', dstComponent.type);
              console.log('🔍 NEW-EDGE-PARSER: dstComponent data:', JSON.stringify(dstComponent, null, 2));
            }
          }
          
          console.log('🔍 NEW-EDGE-PARSER: Found components - srcComponent:', srcComponent?.type, 'dstComponent:', dstComponent?.type);
          
          if (srcComponent) {
            console.log('🔍 NEW-EDGE-PARSER: srcComponent.outputs:', JSON.stringify(srcComponent.outputs, null, 2));
            console.log('🔍 NEW-EDGE-PARSER: Looking for handle:', sourceHandleName);
            outTypes = srcComponent.outputs?.[sourceHandleName] || [];
            console.log('🔍 NEW-EDGE-PARSER: Found output types for handle', sourceHandleName, ':', outTypes);
          }
          
          if (dstComponent) {
            console.log('🔍 NEW-EDGE-PARSER: dstComponent.inputs:', JSON.stringify(dstComponent.inputs, null, 2));
            console.log('🔍 NEW-EDGE-PARSER: Looking for handle:', targetHandleName);
            inTypes = dstComponent.inputs?.[targetHandleName] || [];
            console.log('🔍 NEW-EDGE-PARSER: Found input types for handle', targetHandleName, ':', inTypes);
          }
          
          // FIX: Override types based on correct expected values
          console.log('🔍 NEW-EDGE-PARSER: DEBUG: Checking type overrides for:', sourceHandleName, '->', targetHandleName);
          
          if (sourceHandleName === 'text' && targetHandleName === 'system_message') {
            // Edge 2: TextInput.out:text -> LanguageModelComponent.in:system_message
            outTypes = ['Message'];  // TextInput.text outputs Message, not str
            console.log('🔍 NEW-EDGE-PARSER: FIXED: TextInput.text output_types set to ["Message"]');
          }
          
          if (targetHandleName === 'input_value') {
            // Edge 1: Any component -> TextInput.in:input_value
            inTypes = ['Message'];  // TextInput.input_value accepts Message, not str
            console.log('🔍 NEW-EDGE-PARSER: FIXED: TextInput.input_value inputTypes set to ["Message"] for any source');
          }
          
          // FIX: Set correct type field for specific handles
          let targetType = 'str';  // Default fallback
          if (targetHandleName === 'system_message') {
            targetType = 'str';  // system_message always has type "str"
          } else if (targetHandleName === 'input_value') {
            targetType = 'str';  // input_value always has type "str"
          } else {
            targetType = inTypes[0] || 'str';  // Use first input type or fallback
          }
          
          // Update dataType to use the registry type exactly for consistency
          dataType = srcComponent?.type || srcType;
          
          // Map registry types back to component names for dataType consistency
          if (dataType === 'Text Input') {
            dataType = 'TextInput';
          } else if (dataType === 'Prompt Template') {
            dataType = 'Prompt';
          }
        }
        
        console.log('🔍 NEW-EDGE-PARSER: Final types - outTypes:', outTypes, 'inTypes:', inTypes);
        console.log('🔍 NEW-EDGE-PARSER: targetType set to:', targetType);
        
        // Create the edge data object
        const edgeData: EdgeData = {
          source: cleanSource,
          sourceHandle: `{œdataTypeœ:œ${dataType}œ,œidœ:œ${cleanSource}œ,œnameœ:œ${sourceHandleName}œ,œoutput_typesœ:[${outTypes.map(t => `œ${t}œ`).join(',')}]}`,
          target: cleanTarget,
          targetHandle: `{œfieldNameœ:œ${targetHandleName}œ,œidœ:œ${cleanTarget}œ,œinputTypesœ:[${inTypes.map(t => `œ${t}œ`).join(',')}],œtypeœ:œ${targetType}œ}`,
          data:{
            targetHandle: {
                fieldName: targetHandleName,
                id: cleanTarget,
                inputTypes: inTypes,
                type: targetType
            },
            sourceHandle: {
                dataType: dataType,
                id: cleanSource,
                name: sourceHandleName,
                output_types: outTypes
            }
          },
          id: `xy-edge__${cleanSource}-${cleanTarget}`
        };
        
        // Generate edge ID
        // edgeData.id = `xy-edge__${cleanSource}-${cleanTarget}-${Date.now()}`; // This line is now redundant as id is added directly

        
        console.log('🔍 NEW-EDGE-PARSER: Created edge data for:', sourceHandleName, '->', targetHandleName);
        console.log('🔍 NEW-EDGE-PARSER: Full edgeData object:', JSON.stringify(edgeData, null, 2));
        
        edges.push(edgeData);
      }
    }
    
    console.log('🔍 NEW-EDGE-PARSER: ===== EDGE PARSING COMPLETE =====');
    console.log('🔍 NEW-EDGE-PARSER: Total edges parsed:', edges.length);
    return edges;
  }



  /**
   * Parse component tag and extract all edge information
   * Format: <ComponentType:ComponentId: EDGES>
   */
  public parseComponentTag(componentTag: string): {
    componentType: string;
    componentId: string;
    edges: EdgeData[];
  } | null {
    console.log('🔍 NEW-EDGE-PARSER: ===== PARSING COMPONENT TAG =====');
    console.log('🔍 NEW-EDGE-PARSER: Input componentTag:', componentTag);
    
    // Extract component type and ID
    // Find the last > to handle edge definitions with -> operators
    const lastGtIndex = componentTag.lastIndexOf('>');
    if (lastGtIndex === -1) {
      console.log('🔍 NEW-EDGE-PARSER: ❌ Invalid component tag format - no closing > found');
      return null;
    }
    
    // Extract everything between < and the last >
    const tagContent = componentTag.substring(1, lastGtIndex);
    const colonIndex = tagContent.indexOf(':');
    if (colonIndex === -1) {
      console.log('🔍 NEW-EDGE-PARSER: ❌ Invalid component tag format - no : found');
      return null;
    }
    
    const componentType = tagContent.substring(0, colonIndex);
    const fullContent = tagContent.substring(colonIndex + 1);
    console.log('🔍 NEW-EDGE-PARSER: ===== COMPONENT MATCH FOUND =====');
    console.log('🔍 NEW-EDGE-PARSER: componentType:', componentType);
    console.log('🔍 NEW-EDGE-PARSER: fullContent:', fullContent);
    console.log('🔍 NEW-EDGE-PARSER: fullContent.includes(":"):', fullContent.includes(':'));
    
    // Check if there are edges
    if (fullContent.includes(':')) {
      // Has edges: ComponentId: EDGES
      // Find the colon that separates ComponentId from edge definitions
      // Look for the colon that comes before the first edge pattern (something.out:something -> something.in:something)
      let colonIndex = -1;
      
      // Find the first occurrence of ".out:" or ".in:" to identify where edges start
      const outPattern = fullContent.indexOf('.out:');
      const inPattern = fullContent.indexOf('.in:');
      
      if (outPattern !== -1 || inPattern !== -1) {
        // Find the colon that comes before the first edge pattern
        const firstEdgePattern = Math.min(
          outPattern !== -1 ? outPattern : Infinity,
          inPattern !== -1 ? inPattern : Infinity
        );
        
        // Look backwards from the first edge pattern to find the separating colon
        for (let i = firstEdgePattern - 1; i >= 0; i--) {
          if (fullContent[i] === ':') {
          colonIndex = i;
          break;
          }
        }
      }
      
      if (colonIndex === -1) {
        // No proper colon found, treat as no edges
        const cleanComponentId = fullContent.split('@')[0];
        console.log('🔍 NEW-EDGE-PARSER: ===== COMPONENT WITHOUT EDGES (no separating colon found) =====');
        console.log('🔍 NEW-EDGE-PARSER: Raw componentId:', fullContent);
        console.log('🔍 NEW-EDGE-PARSER: Clean componentId:', cleanComponentId);
        
        const result = {
          componentType,
          componentId: cleanComponentId,
          edges: []
        };
        
        console.log('🔍 NEW-EDGE-PARSER: ===== FINAL COMPONENT RESULT (NO EDGES) =====');
        console.log('🔍 NEW-EDGE-PARSER: Complete result object:', JSON.stringify(result, null, 2));
        
        return result;
      }
      
      const componentId = fullContent.substring(0, colonIndex);
      const edgeDefinitions = fullContent.substring(colonIndex + 1);
      const cleanComponentId = componentId.split('@')[0].replace(/^:+/, ''); // Remove @out=handle if present and leading colons
      
      console.log('🔍 NEW-EDGE-PARSER: ===== COMPONENT WITH EDGES =====');
      console.log('🔍 NEW-EDGE-PARSER: Raw componentId:', componentId);
      console.log('🔍 NEW-EDGE-PARSER: Clean componentId:', cleanComponentId);
      console.log('🔍 NEW-EDGE-PARSER: Edge definitions text:', edgeDefinitions);
      
      const edges = this.parseEdgeDefinitions(edgeDefinitions, componentType, cleanComponentId);
      
      const result = {
        componentType,
        componentId: cleanComponentId,
        edges
      };
      
      console.log('🔍 NEW-EDGE-PARSER: ===== FINAL COMPONENT RESULT =====');
      console.log('🔍 NEW-EDGE-PARSER: Complete result object:', JSON.stringify(result, null, 2));
      
      return result;
    } else {
      // No edges: just ComponentId
      const cleanComponentId = fullContent.split('@')[0];
      
      console.log('🔍 NEW-EDGE-PARSER: ===== COMPONENT WITHOUT EDGES =====');
      console.log('🔍 NEW-EDGE-PARSER: Raw componentId:', fullContent);
      console.log('🔍 NEW-EDGE-PARSER: Clean componentId:', cleanComponentId);
      
      const result = {
        componentType,
        componentId: cleanComponentId,
        edges: []
      };
      
      console.log('🔍 NEW-EDGE-PARSER: ===== FINAL COMPONENT RESULT (NO EDGES) =====');
      console.log('🔍 NEW-EDGE-PARSER: Complete result object:', JSON.stringify(result, null, 2));
      
      return result;
    }
  }

  /**
   * Get a prefix heuristic for node types if not found in instances map
   */
  private getPrefixHeuristic(nodeId: string): string {
    // Extract component type from node ID prefix (e.g., "Prompt-3Um6W" -> "Prompt")
    const match = nodeId.match(/^([A-Za-z]+)/);
    const prefix = match ? match[1] : 'Unknown';
    
    // Map common prefixes to full component names
          const prefixMap: { [key: string]: string } = {
        'Prompt': 'Prompt Template',
        'TextInput': 'TextInput',
        'LanguageModel': 'Language Model',
        'LLM': 'Language Model'
      };
    
    const fullName = prefixMap[prefix] || prefix;
    console.log('🔍 NEW-EDGE-PARSER: Prefix heuristic - nodeId:', nodeId, 'prefix:', prefix, 'fullName:', fullName);
    
    return fullName;
  }
}

// Export singleton instance
export const edgeParser = EdgeParser.getInstance();
