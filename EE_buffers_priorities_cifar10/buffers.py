from collections import deque
import heapq

# defining the basic OOP structures for the buffers:

NUM_PRIORITIES = 3

class FIFO_buffer:
    def __init__(self, max_capacity):
        self.buffer = deque()
        self.max_capacity = max_capacity

    def add_sample(self, sample):
        if len(self.buffer) < self.max_capacity:
            self.buffer.append(sample)
            return True  #  Return success status
        else:
            self.buffer.popleft()
            self.buffer.append(sample)
            return True  #  Still successful, just replaced oldest

    def get_buffer(self):
        return self.buffer
    
    #  Add methods that simulation.py expects
    def get_next_sample(self):
        """Get the next sample to process (FIFO order)"""
        if self.buffer:
            return self.buffer.popleft()
        return None
    
    def get_total_length(self):
        """Get total number of samples in buffer"""
        return len(self.buffer)
    
    def get_buffer_lengths(self):
        """Return buffer lengths for each priority (for compatibility)"""
        return [len(self.buffer), 0, 0]  # Only first priority has items in FIFO


class Strict_Priority_buffer:
    def __init__(self, max_capacity):
        self.buffers = [deque() for _ in range(NUM_PRIORITIES)]
        self.max_capacity = max_capacity

    def add_sample(self, sample, priority=None):
        #  Calculate priority if not provided
        if priority is None:
            priority = self._assign_priority(sample)
        
        #  Priority indexing (convert 1,2,3 to 0,1,2)
        priority_index = priority - 1 if priority > 0 else 0
        priority_index = max(0, min(priority_index, NUM_PRIORITIES - 1))
        
        if len(self.buffers[priority_index]) < self.max_capacity:
            self.buffers[priority_index].append(sample)
            return True
        else:
            return False  #  Return False when buffer is full (sample dropped)
    
    def _assign_priority(self, sample):
        """Assign priority based on confidence (imported from mab_functions logic)"""
        conf = sample['conf_branch_1']
        if conf >= 0.8:
            return 1  # High priority
        elif conf >= 0.6:
            return 2  # Medium priority
        else:
            return 3  # Low priority

    def get_buffers(self):
        return self.buffers
    
    #  Add methods that simulation.py expects
    def get_next_sample(self):
        """Get the next sample to process (strict priority order)"""
        for buffer in self.buffers:
            if buffer:
                return buffer.popleft()
        return None
    
    def get_total_length(self):
        """Get total number of samples across all priority buffers"""
        return sum(len(buffer) for buffer in self.buffers)
    
    def get_buffer_lengths(self):
        """Return buffer lengths for each priority"""
        return [len(buffer) for buffer in self.buffers]


class HybridBuffer:
    def __init__(self, max_capacity):
        self.buffer = []
        self.max_capacity = max_capacity
        self.counter = 0  # for FIFO in case of ties

    def add_sample(self, sample, priority=None):
        #  Calculate priority if not provided
        if priority is None:
            priority = self._assign_priority(sample)
            
        entry = (priority, self.counter, sample)
        self.counter += 1

        if len(self.buffer) < self.max_capacity:
            heapq.heappush(self.buffer, entry)
            return True
        else:
            # find the worst element in the heap
            worst = max(self.buffer, key=lambda x: (x[0], x[1]))

            if priority < worst[0]:  # when the entry sample priority is better than the worst one in the heap
                self.buffer.remove(worst)
                heapq.heapify(self.buffer)
                heapq.heappush(self.buffer, entry)
                return True
            else:
                return False  #  Return False when sample is dropped
    
    def _assign_priority(self, sample):
        """Assign priority based on confidence"""
        conf = sample['conf_branch_1']
        if conf >= 0.8:
            return 1  # High priority (lower number = higher priority in heap)
        elif conf >= 0.6:
            return 2  # Medium priority
        else:
            return 3  # Low priority
    
    #  Add methods that simulation.py expects
    def get_next_sample(self):
        """Get the next sample to process (priority order)"""
        if self.buffer:
            priority, counter, sample = heapq.heappop(self.buffer)
            return sample
        return None
    
    def get_total_length(self):
        """Get total number of samples in buffer"""
        return len(self.buffer)
    
    def get_buffer_lengths(self):
        """Return buffer lengths for each priority (approximation for hybrid)"""
        priority_counts = [0, 0, 0]
        for priority, _, _ in self.buffer:
            if 1 <= priority <= 3:
                priority_counts[priority - 1] += 1
        return priority_counts