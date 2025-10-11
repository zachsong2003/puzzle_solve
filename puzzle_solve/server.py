import os
import base64
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import datetime

# Ensure OPENAI_API_KEY is set in your environment
# export OPENAI_API_KEY=sk-...

client = OpenAI()

# Define puzzle pieces structure
PUZZLE_GRID = {
    "rows": 3,
    "columns": 2,
    "pieces": [
        {"row": 0, "column": 0, "file": "row0_col0.png"},
        {"row": 0, "column": 1, "file": "row0_col1.png"},
        {"row": 1, "column": 0, "file": "row1_col0.png"},
        {"row": 1, "column": 1, "file": "row1_col1.png"},
        {"row": 2, "column": 0, "file": "row2_col0.png"},
        {"row": 2, "column": 1, "file": "row2_col1.png"},
    ]
}

def get_puzzle_pieces_path():
    """Get the path to the puzzle pieces directory"""
    # return Path("terracotta_black_bg2/")
    return Path("terracotta_black_bg2/")

def load_reference_pieces():
    """Load reference puzzle pieces as base64 encoded strings"""
    pieces_path = get_puzzle_pieces_path()
    reference_pieces = {}
    
    for piece in PUZZLE_GRID["pieces"]:
        piece_file = pieces_path / piece["file"]
        if piece_file.exists():
            with open(piece_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                reference_pieces[f"row{piece['row']}_col{piece['column']}"] = encoded
    
    return reference_pieces

app = FastAPI()

# Allow local dev from any origin. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/solve")
async def solve(image: UploadFile = File(...), size: str = Form("1024x1024"), use_reference: str = Form("false")):
    # Read uploaded image
    begin_time = datetime.datetime.now()
    data = await image.read()
    
    # Convert string to boolean
    use_reference_bool = use_reference.lower() == "true"

    print(f"solve size: {size}, use_reference: {use_reference_bool}, raw use_reference: {use_reference}")

    # Wrap bytes in a file-like object and give it a name with extension
    img_buf = BytesIO(data)
    img_buf.name = "puzzle.png"

    # Enhanced prompt with puzzle pieces structure information
    if use_reference_bool:
        prompt = (
            "You are an expert at reconstructing Terracotta Army puzzles from scattered pieces. "
            "The puzzle consists of exactly 6 pieces arranged in a 3x2 grid (3 rows, 2 columns). "
            "The correct arrangement should be:\n"
            "- Row 0: [Top-left piece] [Top-right piece]\n"
            "- Row 1: [Middle-left piece] [Middle-right piece]\n"
            "- Row 2: [Bottom-left piece] [Bottom-right piece]\n\n"
            "Given the provided image showing scrambled puzzle pieces of the Terracotta Army, "
            "reconstruct them into the correct 3x2 grid formation to reveal the complete, coherent scene. "
            "Each piece should fit perfectly with its neighbors, forming seamless edges. "
            "The final image should show the authentic Terracotta Army warriors in their proper formation. "
            "Pay special attention to:\n"
            "1. Proper alignment of piece edges\n"
            "2. Continuity of warrior figures across piece boundaries\n"
            "3. Consistent lighting and perspective\n"
            "4. Authentic Terracotta Army appearance and coloring"
        )
    else:
        # Fallback to original prompt
        prompt = (
            "You are an expert at reconstructing Terracotta Army from pieces. "
            "Given the provided image showing puzzle pieces of Terracotta Army, "
            "produce a single coherent, solved image that faithfully matches the original scene. "
            "Remember, you are exactly constructing Terracotta Army, not any other kind of things."
        )

    print(prompt)
    try:
        # Use the Images Edit API with the input image as the edit base.
        # Note: We do not provide a mask so the model can transform the full image.
        result = client.images.edit(
            model="gpt-image-1",
            image=[img_buf],
            prompt=prompt,
            size=size,         # e.g., "1024x1024"
            n=1,
        )
        b64 = result.data[0].b64_json
        print(f"spend time: {datetime.datetime.now() - begin_time}")
        return JSONResponse({
            "image_b64": b64,
            "grid_info": PUZZLE_GRID if use_reference_bool else None,
            "prompt_used": "enhanced" if use_reference_bool else "basic"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/puzzle-info")
async def get_puzzle_info():
    """Get information about the puzzle structure and available reference pieces"""
    pieces_path = get_puzzle_pieces_path()
    available_pieces = []
    
    for piece in PUZZLE_GRID["pieces"]:
        piece_file = pieces_path / piece["file"]
        available_pieces.append({
            "row": piece["row"],
            "column": piece["column"],
            "filename": piece["file"],
            "exists": piece_file.exists()
        })
    
    return JSONResponse({
        "grid_structure": PUZZLE_GRID,
        "available_pieces": available_pieces,
        "pieces_directory": str(pieces_path)
    })


@app.post("/solve-with-references")
async def solve_with_references(image: UploadFile = File(...), size: str = "1024x1024"):
    """Enhanced solve endpoint that uses reference pieces as visual context"""
    # Read uploaded image
    print(f"solve-with-references size: {size}")
    data = await image.read()
    scrambled_b64 = base64.b64encode(data).decode('utf-8')
    
    # Load reference pieces
    reference_pieces = load_reference_pieces()
    
    if not reference_pieces:
        return JSONResponse({"error": "No reference pieces found"}, status_code=400)
    
    # Create messages with multiple images for vision analysis
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I have a Terracotta Army puzzle with 6 pieces that need to be assembled into a 3x2 grid. "
                        "The first image shows the scrambled pieces on a surface. "
                        "The following 6 images show the correct reference pieces in their proper positions:\n"
                        "- Reference 1: row0_col0 (top-left)\n"
                        "- Reference 2: row0_col1 (top-right)\n"
                        "- Reference 3: row1_col0 (middle-left)\n"
                        "- Reference 4: row1_col1 (middle-right)\n"
                        "- Reference 5: row2_col0 (bottom-left)\n"
                        "- Reference 6: row2_col1 (bottom-right)\n\n"
                        "Please analyze the scrambled pieces and identify which piece corresponds to which reference position. "
                        "Then provide detailed instructions on how to arrange them to form the complete Terracotta Army scene."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{scrambled_b64}"}
                }
            ]
        }
    ]
    
    # Add reference pieces to the message
    reference_labels = [
        "row0_col0", "row0_col1", "row1_col0",
        "row1_col1", "row2_col0", "row2_col1"
    ]
    
    for label in reference_labels:
        if label in reference_pieces:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{reference_pieces[label]}"}
            })

    try:
        # First, analyze the pieces using vision
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            timeout=240,
        )
        
        analysis = vision_response.choices[0].message.content
        
        # Now use the analysis to create a better reconstruction prompt
        img_buf = BytesIO(data)
        img_buf.name = "puzzle.png"
        
        reconstruction_prompt = (
            f"Based on this analysis: {analysis}\n\n"
            "Now reconstruct the Terracotta Army puzzle by arranging the scrambled pieces into the correct 3x2 grid formation. "
            "Each piece should be placed in its identified position to form a seamless, complete image of Terracotta Army warriors. "
            "Ensure proper alignment, consistent lighting, and authentic historical appearance."
        )
        
        # Use image edit API for reconstruction
        result = client.images.edit(
            model="gpt-image-1",
            image=[img_buf],
            prompt=reconstruction_prompt,
            size=size,
            n=1,
        )
        
        return JSONResponse({
            "image_b64": result.data[0].b64_json,
            "analysis": analysis,
            "reference_pieces_used": len(reference_pieces),
            "grid_info": PUZZLE_GRID,
            "method": "vision_analysis_plus_reconstruction"
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/analyze-pieces")
async def analyze_pieces(image: UploadFile = File(...)):
    """Analyze scrambled puzzle pieces and identify their positions using reference pieces"""
    # Read uploaded image
    print("analyze-pieces")
    data = await image.read()
    scrambled_b64 = base64.b64encode(data).decode('utf-8')
    
    # Load reference pieces
    reference_pieces = load_reference_pieces()
    
    if not reference_pieces:
        return JSONResponse({"error": "No reference pieces found"}, status_code=400)
    
    # Create messages for vision analysis
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I have 6 Terracotta Army puzzle pieces scattered on a surface (first image). "
                        "I also have 6 reference images showing what each piece should look like in the correct positions:\n"
                        "- Reference 1: row0_col0 (top-left position)\n"
                        "- Reference 2: row0_col1 (top-right position)\n"
                        "- Reference 3: row1_col0 (middle-left position)\n"
                        "- Reference 4: row1_col1 (middle-right position)\n"
                        "- Reference 5: row2_col0 (bottom-left position)\n"
                        "- Reference 6: row2_col1 (bottom-right position)\n\n"
                        "Please identify each scattered piece in the first image and match it to its correct reference position. "
                        "The output mapping is in json format. "
                        "The key is the positon of the scattered piece, e.g. scattered_row0_col0. "
                        "The value is the position of the reference piece, e.g. reference_row0_col0."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{scrambled_b64}"}
                }
            ]
        }
    ]
    
    # Add reference pieces
    reference_labels = [
        "row0_col0", "row0_col1", "row1_col0",
        "row1_col1", "row2_col0", "row2_col1"
    ]
    
    for label in reference_labels:
        if label in reference_pieces:
            print(f"label: {label}, reference: {label in reference_pieces}")
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{reference_pieces[label]}"}
            })

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        return JSONResponse({
            "analysis": response.choices[0].message.content,
            "reference_pieces_used": len(reference_pieces),
            "grid_info": PUZZLE_GRID
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)