import gurobipy as gp
from gurobipy import GRB
import json
import sys

def solve_bwm(aB, aW, best_index, worst_index):
    """
    Solve the BWM optimization problem using Gurobi.
    Implements the linear BWM formulation exactly as described in the original paper.
    """
    try:
        # Create a new model
        model = gp.Model("BWM")
        model.setParam('OutputFlag', 0)
        
        # Create variables
        n = len(aB)
        w = model.addVars(n, lb=0.01, ub=1.0, name="w")  # weights
        xi = model.addVar(name="xi")    # consistency variable
        
        # Set objective: minimize maximum absolute difference
        model.setObjective(xi, GRB.MINIMIZE)
        
        # Constraints
        
        # Sum of weights = 1
        model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1)
        
        # Best-to-others constraints
        for j in range(n):
            if j != best_index:
                # |w_B/w_j - a_Bj| ≤ xi
                model.addConstr((w[best_index] - aB[j] * w[j]) <= xi)
                model.addConstr((aB[j] * w[j] - w[best_index]) <= xi)
        
        # Others-to-worst constraints
        for i in range(n):
            if i != worst_index:
                # |w_i/w_W - a_iW| ≤ xi
                model.addConstr((w[i] - aW[i] * w[worst_index]) <= xi)
                model.addConstr((aW[i] * w[worst_index] - w[i]) <= xi)
        
        # Optimize
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Get optimal weights
            weights = [w[i].X for i in range(n)]
            
            # Get optimal xi
            optimal_xi = xi.X
            
            # Calculate consistency ratio
            max_aBj = max(aB)
            
            # Consistency Index table from the BWM paper
            ci_table = {
                1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30,
                6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23
            }
            
            # Get consistency index from table
            ci = ci_table.get(max_aBj, 5.23)  # use 5.23 for values > 9
            
            # Calculate consistency ratio
            consistency_ratio = optimal_xi / ci if ci > 0 else 0
            
            # Print debug info
            print(f"Input vectors:")
            print(f"Best-to-others (aB): {aB}")
            print(f"Others-to-worst (aW): {aW}")
            print(f"Best index: {best_index}")
            print(f"Worst index: {worst_index}")
            print(f"Results:")
            print(f"Weights: {[round(w, 3) for w in weights]}")
            print(f"Optimal xi: {round(optimal_xi, 3)}")
            print(f"Consistency ratio: {round(consistency_ratio, 3)}")
            
            return weights, consistency_ratio
            
        else:
            raise Exception("Optimization failed")
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    aB = input_data["aB"]
    aW = input_data["aW"]
    best_index = input_data["best_index"]
    worst_index = input_data["worst_index"]
    
    # Solve the problem
    weights, consistency_ratio = solve_bwm(aB, aW, best_index, worst_index)
    
    # Prepare output
    output = {
        "weights": weights,
        "consistency_ratio": consistency_ratio
    }
    
    # Write output to stdout
    print(json.dumps(output)) 