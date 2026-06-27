from collections import OrderedDict


class ARC:
    def __init__(self, capacity):
        self.c = capacity
        self.p = 0  # adaptive target size for T1

        # Cache lists
        self.T1 = OrderedDict()  # recent
        self.T2 = OrderedDict()  # frequent

        # Ghost lists
        self.B1 = OrderedDict()
        self.B2 = OrderedDict()

        self.hits = 0
        self.misses = 0

    def _move_to_mru(self, lst, key):
        """Move key to MRU position."""
        lst.move_to_end(key)

    def _add_mru(self, lst, key):
        lst[key] = None
        lst.move_to_end(key)

    def _remove_lru(self, lst):
        """Remove and return LRU item."""
        return lst.popitem(last=False)[0]

    def _replace(self, x):
        """
        REPLACE(x, p)

        If (|T1| > p) or (x in B2 and |T1| == p)
            move LRU(T1) -> MRU(B1)
        else
            move LRU(T2) -> MRU(B2)
        """

        if (
            len(self.T1) > 0
            and (
                len(self.T1) > self.p
                or (x in self.B2 and len(self.T1) == self.p)
            )
        ):
            victim = self._remove_lru(self.T1)
            self._add_mru(self.B1, victim)

        else:
            victim = self._remove_lru(self.T2)
            self._add_mru(self.B2, victim)

    def request(self, x):
        """
        Process one request x.
        Returns True on hit, False on miss.
        """

        # --------------------------------------------------
        # CASE I: hit in T1 or T2
        # --------------------------------------------------
        if x in self.T1 or x in self.T2:
            self.hits += 1

            if x in self.T1:
                del self.T1[x]
            else:
                del self.T2[x]

            self._add_mru(self.T2, x)
            return True

        # --------------------------------------------------
        # CASE II: hit in B1
        # --------------------------------------------------
        if x in self.B1:
            self.misses += 1

            if len(self.B1) >= len(self.B2):
                delta = 1
            else:
                delta = len(self.B2) / len(self.B1)

            self.p = min(self.c, self.p + delta)

            self._replace(x)

            del self.B1[x]
            self._add_mru(self.T2, x)

            return False

        # --------------------------------------------------
        # CASE III: hit in B2
        # --------------------------------------------------
        if x in self.B2:
            self.misses += 1

            if len(self.B2) >= len(self.B1):
                delta = 1
            else:
                delta = len(self.B1) / len(self.B2)

            self.p = max(0, self.p - delta)

            self._replace(x)

            del self.B2[x]
            self._add_mru(self.T2, x)

            return False

        # --------------------------------------------------
        # CASE IV: completely new page
        # --------------------------------------------------
        self.misses += 1

        l1_size = len(self.T1) + len(self.B1)

        # ---------- Case A ----------
        if l1_size == self.c:

            if len(self.T1) < self.c:
                # Remove LRU(B1)
                self._remove_lru(self.B1)

                self._replace(x)

            else:
                # B1 empty
                self._remove_lru(self.T1)

        # ---------- Case B ----------
        elif l1_size < self.c:

            total = (
                len(self.T1)
                + len(self.T2)
                + len(self.B1)
                + len(self.B2)
            )

            if total >= self.c:

                if total == 2 * self.c:
                    self._remove_lru(self.B2)

                self._replace(x)

        # Insert new page into T1
        self._add_mru(self.T1, x)

        return False

    @property
    def cache(self):
        return list(self.T1.keys()) + list(self.T2.keys())

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "p": self.p,
        }