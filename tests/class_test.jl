using ObjectOriented

@oodef mutable struct MyClass 
    a::Int
    b::Int

    function new(a::Int, b::Int)
        self = @mk
        self.a = a
        self.b = b
        return self
    end
    function compute_a_plus_b(self)
        self.a + self.b
    end
end

inst = MyClass(1, 2)
print(inst.compute_a_plus_b())
