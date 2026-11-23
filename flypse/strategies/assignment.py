"""Module for the assignment strategies.

Assignment decides what each selected entity is asked to do: instructions sent
along with the model, as the data of the message. E.g., train for 5 epochs,
answer within 2 seconds (FedCS), or train only a half-width sub-model because
the device is weak (HeteroFL).

It is apart from [selection][flypse.strategies.selection] because the two vary
independently: e.g., the selection of Oort works with or without the sub-models of
HeteroFL.

No strategy is here yet: a fixed budget or deadline is just data, e.g.
``await self.send(selected, record, epochs=5)``.
"""
