import Mapping

Current_map = [['\0' for _ in range(550)] for _ in range(100)]
Predicted_map = [['\0' for _ in range(550)] for _ in range(100)]

def main():
    Mapping.scan_map(Current_map)

    
