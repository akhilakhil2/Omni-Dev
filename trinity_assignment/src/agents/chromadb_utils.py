# utils/chromadb_utils.py

def build_chromadb_filter(conditions: list, logic="or") -> dict:
    """
    Build ChromaDB filter safely.
    - Single condition → return it directly
    - Multiple conditions → wrap in $and/$or
    """
    if not conditions:
        return {}
    
    if len(conditions) == 1:
        return conditions[0]

    if logic == "or":
        return {"$or": conditions}
    elif logic == "and":
        return {"$and": conditions}
    else:
        raise ValueError("logic must be 'and' or 'or'")
