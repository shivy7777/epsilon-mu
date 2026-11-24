"""Reserve estimated resources before dispatch; settle each attempt once."""
import math
import time
from decimal import Decimal
from threading import Lock

from .configuration import AIError


class Budget:
    def __init__(self, max_attempts=4, max_tokens=None, max_cost=None,
                 rates=None, deadline_seconds=None, clock=time.monotonic):
        if type(max_attempts)!=int or max_attempts<1 or (max_tokens is not None and
                (type(max_tokens)!=int or max_tokens<1)):
            raise AIError('Invalid investigation limit.')
        if deadline_seconds is not None and (not math.isfinite(deadline_seconds) or deadline_seconds<=0):
            raise AIError('Invalid investigation deadline.')
        self.clock=clock
        self.deadline=None if deadline_seconds is None else clock()+deadline_seconds
        self.max_attempts=max_attempts
        self.max_tokens=max_tokens
        self.max_cost=None if max_cost is None else Decimal(str(max_cost))
        self.rates=None if rates is None else tuple(Decimal(str(v)) for v in rates)
        if self.rates is not None and (len(self.rates)!=2 or any(not v.is_finite() or v<0 for v in self.rates)):
            raise AIError('Supply two finite nonnegative rates per million tokens.')
        if self.max_cost is not None and (not self.max_cost.is_finite() or self.max_cost<0 or self.rates is None):
            raise AIError('A cost limit requires nonnegative prices and a finite limit.')
        self.attempts=self.successful=self.rejected=self.failed=self.unknown_usage=0
        self.reported_input_tokens=self.reported_output_tokens=0
        self.charged_tokens=0
        self.estimated_cost=None if self.rates is None else Decimal(0)
        self.pending={}
        self.lock=Lock()

    def remaining(self):
        remaining=math.inf if self.deadline is None else self.deadline-self.clock()
        if remaining<=0:
            raise AIError('Investigation deadline reached.')
        return remaining

    def _cost(self,inputs,outputs):
        if self.rates is None:
            return Decimal(0)
        return (inputs*self.rates[0]+outputs*self.rates[1])/1_000_000

    def reserve(self,inputs,outputs):
        with self.lock:
            self.remaining()
            cost=self._cost(inputs,outputs)
            if (self.attempts>=self.max_attempts or
                    (self.max_tokens is not None and self.charged_tokens+inputs+outputs>self.max_tokens) or
                    (self.max_cost is not None and self.estimated_cost+cost>self.max_cost)):
                raise AIError('Investigation budget exhausted.')
            self.attempts+=1
            self.pending[self.attempts]=(inputs,outputs)
            self.charged_tokens+=inputs+outputs
            if self.estimated_cost is not None:
                self.estimated_cost+=cost
            return self.attempts

    def settle(self,ticket,usage,outcome):
        with self.lock:
            reserved=self.pending.pop(ticket)
            inputs,outputs=usage
            self.unknown_usage+=int(None in usage)
            self.reported_input_tokens+=inputs or 0
            self.reported_output_tokens+=outputs or 0
            actual=tuple(r if u is None else u for r,u in zip(reserved,usage))
            self.charged_tokens+=sum(actual)-sum(reserved)
            if self.estimated_cost is not None:
                self.estimated_cost+=self._cost(*actual)-self._cost(*reserved)
            setattr(self,outcome,getattr(self,outcome)+1)
