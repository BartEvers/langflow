import pytest
from unittest.mock import Mock, patch

from langflow.airelius.service import PFUService


class TestPFUService:
    def test_execute_plan_step_by_step_success(self):
        """Test successful step-by-step execution of a PFU plan."""
        service = PFUService()
        
        # Mock plan with simple operations
        plan = {
            "operations": [
                {
                    "step": 1,
                    "description": "Add a test node",
                    "validation": "Check if node was added",
                    "operation": {
                        "op": "add_node",
                        "node": {
                            "id": "test-node-1",
                            "type": "genericNode",
                            "position": {"x": 100, "y": 100},
                            "data": {"id": "test-node-1", "type": "TestComponent"}
                        }
                    }
                }
            ]
        }
        
        # Initial flow data
        initial_flow = {"nodes": [], "edges": []}
        
        # Mock the _apply_flow_operations function
        with patch('langflow.airelius.service._apply_flow_operations') as mock_apply:
            mock_apply.return_value = {
                "nodes": [{"id": "test-node-1", "type": "genericNode"}],
                "edges": []
            }
            
            result = service.execute_plan_step_by_step(plan, initial_flow)
        
        assert result["execution_summary"]["total_steps"] == 1
        assert result["execution_summary"]["successful_steps"] == 1
        assert result["execution_summary"]["failed_steps"] == 0
        assert len(result["execution_summary"]["executed_steps"]) == 1
        assert result["execution_summary"]["executed_steps"][0]["status"] == "success"

    def test_execute_plan_step_by_step_with_failures(self):
        """Test execution with some failed steps."""
        service = PFUService()
        
        plan = {
            "operations": [
                {
                    "step": 1,
                    "description": "Add first node",
                    "operation": {
                        "op": "add_node",
                        "node": {"id": "node-1", "type": "genericNode"}
                    }
                },
                {
                    "step": 2,
                    "description": "Add second node (will fail)",
                    "operation": {
                        "op": "add_node",
                        "node": {"id": "node-2", "type": "genericNode"}
                    }
                }
            ]
        }
        
        initial_flow = {"nodes": [], "edges": []}
        
        # Mock _apply_flow_operations to fail on second operation
        def mock_apply(flow_data, ops):
            if ops[0]["op"] == "add_node" and ops[0]["node"]["id"] == "node-2":
                raise ValueError("Simulated failure")
            return {
                "nodes": [{"id": ops[0]["node"]["id"], "type": "genericNode"}],
                "edges": []
            }
        
        with patch('langflow.airelius.service._apply_flow_operations', side_effect=mock_apply):
            result = service.execute_plan_step_by_step(plan, initial_flow)
        
        assert result["execution_summary"]["total_steps"] == 2
        assert result["execution_summary"]["successful_steps"] == 1
        assert result["execution_summary"]["failed_steps"] == 1
        assert len(result["execution_summary"]["failed_steps"]) == 1
        assert result["execution_summary"]["failed_steps"][0]["error"] == "Simulated failure"

    def test_execute_plan_step_by_step_max_steps_limit(self):
        """Test that execution respects the max_steps limit."""
        service = PFUService()
        
        # Create a plan with more operations than max_steps
        plan = {
            "operations": [
                {
                    "step": i,
                    "description": f"Step {i}",
                    "operation": {
                        "op": "add_node",
                        "node": {"id": f"node-{i}", "type": "genericNode"}
                    }
                }
                for i in range(1, 16)  # 15 operations
            ]
        }
        
        initial_flow = {"nodes": [], "edges": []}
        
        with patch('langflow.airelius.service._apply_flow_operations') as mock_apply:
            mock_apply.return_value = {"nodes": [], "edges": []}
            
            result = service.execute_plan_step_by_step(plan, initial_flow, max_steps=10)
        
        # Should only execute 10 steps due to max_steps limit
        assert result["execution_summary"]["total_steps"] == 10
        assert result["execution_summary"]["successful_steps"] == 10

    def test_execute_plan_step_by_step_invalid_plan(self):
        """Test execution with invalid plan structure."""
        service = PFUService()
        
        # Plan missing operations
        invalid_plan = {"objective": "Test"}
        initial_flow = {"nodes": [], "edges": []}
        
        with pytest.raises(ValueError, match="Invalid plan: missing operations"):
            service.execute_plan_step_by_step(invalid_plan, initial_flow)
        
        # Empty operations list
        empty_plan = {"operations": []}
        result = service.execute_plan_step_by_step(empty_plan, initial_flow)
        assert result["execution_summary"]["total_steps"] == 0
        assert result["execution_summary"]["successful_steps"] == 0

    def test_execute_plan_step_by_step_missing_operation(self):
        """Test execution with steps missing operations."""
        service = PFUService()
        
        plan = {
            "operations": [
                {
                    "step": 1,
                    "description": "Step without operation",
                    # Missing operation field
                }
            ]
        }
        
        initial_flow = {"nodes": [], "edges": []}
        
        with patch('langflow.airelius.service._apply_flow_operations') as mock_apply:
            mock_apply.return_value = {"nodes": [], "edges": []}
            
            result = service.execute_plan_step_by_step(plan, initial_flow)
        
        # Should skip the step with missing operation
        assert result["execution_summary"]["total_steps"] == 1
        assert result["execution_summary"]["successful_steps"] == 0
        assert result["execution_summary"]["failed_steps"] == 0
