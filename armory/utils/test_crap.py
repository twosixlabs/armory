from pydantic import BaseModel
from pydantic.utils import GetterDict
from typing import Any, Optional, Dict, Union

from typing import Dict
from pydoc import locate

from pydantic import BaseModel


class ArbitraryValue(BaseModel):
    Value: str
    Type: str


def get_typed_value(thing: ArbitraryValue):
    tp = locate(thing.Type)
    return tp(thing.Value)

class ArbitraryDict(BaseModel):
    __root__: Dict[str, ArbitraryValue]

    def __getattr__(self, item):  # if you want to use '.'
        if item in self.__root__:
            return get_typed_value(self.__root__[item])

    class Config:
        json_encoders = {
            ArbitraryValue: lambda d: get_typed_value(d)

        }




class MyThing(BaseModel):
    name: str
    data: ArbitraryDict

    class Config:
        smart_union = True


class OuterThing(BaseModel):
    thing: MyThing

    class Config:
        smart_union = True

good_data = {
  'a': { 'Value': 'va', 'Type': 'str' },
  'b': { 'Value': '1', 'Type': 'int' },
}

bad_data = {
  'a': { 'Value': 'va', 'Type': 'ta' },
  'b': { 'Value': 'vb' },
}

x = ArbitraryDict.parse_obj(good_data)
print(f"first x: {x}")
print(f"x.a {x.a} type: {type(x.a)}")
print(f"x.b {x.b} type: {type(x.b)}")


y = MyThing.parse_obj({'name':'mystuff','data':good_data})
print(y)
print(f"y data b value: {y.data.b}, type: {type(y.data.b)}")

z = OuterThing.parse_obj({'thing': {'name':'mystuff','data':good_data}})
print(z)
print(f"z data b value: {z.thing.data.b}, type: {type(z.thing.data.b)}")

print(f"{z.dict()}\n\n")
print(f"z.thing.data: {z.thing.data.dict()}")

print(z.json(models_as_dict=False))
# ok !
#
# x = ArbitraryDict.parse_obj(bad_data)
# print(f"second x: {x}")

#
#
# class MyObject(BaseModel):
#     x: str
#     d: Dict[str, Union[str,int]]
#
#     class Config:
#         smart_union = True
#
#
# if __name__ == "__main__":
#     data = {
#         'x': 'seth',
#         'd': {'a': True}
#     }
#
#     obj = MyObject.parse_obj(data)
#     print(obj)
#     print(type(obj.d))
#     print(obj.d.a)
#
