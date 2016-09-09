namespace java cntk

enum Type {
    None,
    Bool,
    SizeT,
    Float,
    Double,
    String,
    NDShape,
    Axis,
    Vector,
    Dictionary,
    NDArrayView,
}

enum DType {
    Unknown,
    Float,
    Double,
}
union Value {
  1: bool bool_value,
  2: i64 sizet_value,
  3: double float_value,
  4: double double_value,
  5: string string_value,
  6: NDShape shape,
  7: Axis axis,
  8: Vector vector,
  9: Dictionary dict,
  10: NDArrayView view,
}

struct NDShape {
  1: list<i64> dims
}

struct Axis {
  1: i64 staticAxisIndex,
  2: string name,
  3: bool is_ordered,
}

struct NDArrayView {
  1: DType type,
  2: NDShape shape,
  3: list<double> elements, // this can be either double or float
}

struct Vector {
  1: list<DictionaryValue> vector,
}

struct DictionaryValue {
    1: Type type,
    2: Value value,
}

struct Dictionary {
  1: map<string, DictionaryValue> dict,
}