# FastAPI Documentation

## What is FastAPI
FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It is one of the fastest Python frameworks available.

## Installation
To install FastAPI run the following command:
pip install fastapi
pip install uvicorn

## Creating Your First App
Create a file main.py with the following content:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

To run the app use this command:
uvicorn main:app --reload

## Path Parameters
You can declare path parameters with the same syntax used by Python format strings:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

## Query Parameters
When you declare function parameters that are not part of the path parameters they are automatically interpreted as query parameters:

```python
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

## Request Body
To declare a request body use Pydantic models:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
def create_item(item: Item):
    return item
```

## Authentication with OAuth2
FastAPI provides OAuth2 with Password Bearer token authentication:

```python
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```

## JWT Token Authentication
To implement full JWT authentication install python-jose:
pip install python-jose passlib

```python
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

## Error Handling
Use HTTPException to return HTTP errors:

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

## Middleware
Add middleware to your FastAPI application:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Dependencies
FastAPI has a dependency injection system:

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
def read_users(db = Depends(get_db)):
    return db.query(User).all()
```