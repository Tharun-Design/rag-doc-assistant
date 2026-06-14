# Pydantic Documentation

## What is Pydantic
Pydantic is a data validation library for Python. It uses Python type annotations to validate data and serialize/deserialize objects.

## Installation
pip install pydantic

## Basic Models
Define a model by inheriting from BaseModel:

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int = 18
```

## Field Validation
Use Field for advanced validation:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price must be positive")
    quantity: int = Field(default=0, ge=0)
```

## Validators
Add custom validators using field_validator:

```python
from pydantic import BaseModel, field_validator

class UserModel(BaseModel):
    username: str
    email: str

    @field_validator("email")
    def email_must_contain_at(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v
```

## Nested Models
Models can be nested inside each other:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    address: Address
```

## Model Config
Configure model behavior using model_config:

```python
from pydantic import BaseModel
from pydantic.config import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )
    name: str
```

## Serialization
Convert models to dict or JSON:

```python
user = User(id=1, name="Sara", email="sara@example.com")

# To dictionary
user_dict = user.model_dump()

# To JSON string
user_json = user.model_dump_json()
```

## Error Handling
Pydantic raises ValidationError when data is invalid:

```python
from pydantic import ValidationError

try:
    user = User(id="not-an-int", name="Sara")
except ValidationError as e:
    print(e.errors())
```

## Optional Fields
Use Optional for fields that can be None:

```python
from typing import Optional
from pydantic import BaseModel

class Profile(BaseModel):
    name: str
    bio: Optional[str] = None
    website: Optional[str] = None
```