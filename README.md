# conversational-rag-app

conversational-rag-app/
│
├── backend/
│   ├── db/                     # ChromaDB will store its data here
│   ├── .env                    # Your secret API keys
│   ├── main.py                 # The main FastAPI application file
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── index.html              # The main UI page
│   ├── script.js               # JavaScript for interactivity and API calls
│   └── style.css               # Styles for the UI
│
└── README.md                   # Instructions on how to set up and run


python3 -m http.server 8080

python -m SimpleHTTPServer 8080

http://localhost:8080

uvicorn main:app --reload
