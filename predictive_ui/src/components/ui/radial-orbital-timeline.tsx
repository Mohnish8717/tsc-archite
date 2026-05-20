"use client";
import { useState, useEffect, useRef } from "react";
import { ArrowRight, Link, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TimelineItem {
  id: number;
  title: string;
  date: string;
  content: string;
  category: string;
  icon: React.ElementType;
  relatedIds: number[];
  status: "completed" | "in-progress" | "pending";
  energy: number;
}

interface RadialOrbitalTimelineProps {
  timelineData: TimelineItem[];
}

export default function RadialOrbitalTimeline({
  timelineData,
}: RadialOrbitalTimelineProps) {
  const [expandedItems, setExpandedItems] = useState<Record<number, boolean>>(
    {}
  );
  const [, setViewMode] = useState<string>("orbital");
  const viewMode = "orbital";
  const [rotationAngle, setRotationAngle] = useState<number>(0);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [pulseEffect, setPulseEffect] = useState<Record<number, boolean>>({});
  const [centerOffset] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  });
  const [activeNodeId, setActiveNodeId] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const orbitRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<Record<number, HTMLDivElement | null>>({});

  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === containerRef.current || e.target === orbitRef.current) {
      setExpandedItems({});
      setActiveNodeId(null);
      setPulseEffect({});
      setAutoRotate(true);
    }
  };

  const toggleItem = (id: number) => {
    setExpandedItems((prev) => {
      const newState = { ...prev };
      Object.keys(newState).forEach((key) => {
        if (parseInt(key) !== id) {
          newState[parseInt(key)] = false;
        }
      });

      newState[id] = !prev[id];

      if (!prev[id]) {
        setActiveNodeId(id);
        setAutoRotate(false);

        const relatedItems = getRelatedItems(id);
        const newPulseEffect: Record<number, boolean> = {};
        relatedItems.forEach((relId) => {
          newPulseEffect[relId] = true;
        });
        setPulseEffect(newPulseEffect);

        centerViewOnNode(id);
      } else {
        setActiveNodeId(null);
        setAutoRotate(true);
        setPulseEffect({});
      }

      return newState;
    });
  };

  useEffect(() => {
    let rotationTimer: ReturnType<typeof setInterval>;

    if (autoRotate && viewMode === "orbital") {
      rotationTimer = setInterval(() => {
        setRotationAngle((prev) => {
          const newAngle = (prev + 0.3) % 360;
          return Number(newAngle.toFixed(3));
        });
      }, 50);
    }

    return () => {
      if (rotationTimer) {
        clearInterval(rotationTimer);
      }
    };
  }, [autoRotate, viewMode]);

  const centerViewOnNode = (nodeId: number) => {
    if (viewMode !== "orbital" || !nodeRefs.current[nodeId]) return;

    const nodeIndex = timelineData.findIndex((item) => item.id === nodeId);
    const totalNodes = timelineData.length;
    const targetAngle = (nodeIndex / totalNodes) * 360;

    setRotationAngle(270 - targetAngle);
  };

  const calculateNodePosition = (index: number, total: number) => {
    const angle = ((index / total) * 360 + rotationAngle) % 360;
    const radius = typeof window !== "undefined" && window.innerWidth < 768 ? 160 : 300;
    const radian = (angle * Math.PI) / 180;

    const x = radius * Math.cos(radian) + centerOffset.x;
    const y = radius * Math.sin(radian) + centerOffset.y;

    const zIndex = Math.round(100 + 50 * Math.cos(radian));
    const opacity = 1; // Solid opacity for brutalist aesthetic

    return { x, y, angle, zIndex, opacity };
  };

  const getRelatedItems = (itemId: number): number[] => {
    const currentItem = timelineData.find((item) => item.id === itemId);
    return currentItem ? currentItem.relatedIds : [];
  };

  const isRelatedToActive = (itemId: number): boolean => {
    if (!activeNodeId) return false;
    const relatedItems = getRelatedItems(activeNodeId);
    return relatedItems.includes(itemId);
  };

  const getStatusStyles = (status: TimelineItem["status"]): string => {
    switch (status) {
      case "completed":
        return "text-white bg-black";
      case "in-progress":
        return "text-black bg-white";
      case "pending":
        return "text-black bg-gray-300";
      default:
        return "text-black bg-white";
    }
  };

  // suppress unused setter warning
  void setViewMode;

  return (
    <div
      className="w-full h-[700px] md:h-screen flex flex-col items-center justify-center bg-transparent overflow-hidden"
      ref={containerRef}
      onClick={handleContainerClick}
    >
      <div className="relative w-full max-w-4xl h-full flex items-center justify-center">
        <div
          className="absolute w-full h-full flex items-center justify-center"
          ref={orbitRef}
          style={{
            perspective: "1000px",
            transform: `translate(${centerOffset.x}px, ${centerOffset.y}px)`,
          }}
        >
          <div className="absolute w-24 h-24 bg-white border-8 border-black flex items-center justify-center z-10 shadow-neo-black transform rotate-3">
            <div className="w-16 h-16 bg-brand border-4 border-black flex items-center justify-center animate-pulse">
                <Zap size={32} className="text-black" strokeWidth={3} />
            </div>
          </div>

          <div className="absolute w-[320px] h-[320px] md:w-[600px] md:h-[600px] rounded-full border-4 border-black border-dashed opacity-40 animate-[spin_60s_linear_infinite]"></div>

          {timelineData.map((item, index) => {
            const position = calculateNodePosition(index, timelineData.length);
            const isExpanded = expandedItems[item.id];
            const isRelated = isRelatedToActive(item.id);
            const isPulsing = pulseEffect[item.id];
            const Icon = item.icon;

            const nodeStyle = {
              transform: `translate(${position.x}px, ${position.y}px)`,
              zIndex: isExpanded ? 200 : position.zIndex,
              opacity: isExpanded ? 1 : position.opacity,
            };

            return (
              <div
                key={item.id}
                ref={(el) => { nodeRefs.current[item.id] = el; }}
                className="absolute transition-all duration-700 cursor-pointer"
                style={nodeStyle}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleItem(item.id);
                }}
              >
                {isPulsing && (
                  <div className="absolute -inset-4 bg-black border-4 border-black rounded-full opacity-20 animate-ping"></div>
                )}

                <div
                  className={`
                  w-14 h-14 flex items-center justify-center border-4 border-black shadow-neo-black transition-all duration-300 transform
                  ${
                    isExpanded
                      ? "bg-black text-white scale-125 -rotate-6 shadow-neo-pressed"
                      : isRelated
                      ? "bg-brand text-black hover:-translate-y-1 hover:translate-x-1 hover:shadow-neo-pressed"
                      : "bg-white text-black hover:-translate-y-1 hover:translate-x-1 hover:shadow-neo-pressed"
                  }
                `}
                >
                  <Icon size={24} strokeWidth={3} />
                </div>

                <div
                  className={`
                  absolute top-16 left-1/2 -translate-x-1/2 whitespace-nowrap
                  font-black uppercase tracking-widest px-2 py-1 border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]
                  transition-all duration-300
                  ${isExpanded ? "bg-black text-white scale-110" : "bg-white text-black"}
                `}
                >
                  {item.title}
                </div>

                {isExpanded && (
                  <Card className="absolute top-24 left-1/2 -translate-x-1/2 w-80 bg-white border-8 border-black shadow-neo-black rounded-none overflow-visible z-50">
                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 w-4 h-6 bg-black"></div>
                    <CardHeader className="pb-4 border-b-8 border-black bg-brand">
                      <div className="flex justify-between items-center mb-2">
                        <Badge
                          className={`px-3 py-1 text-xs font-black uppercase tracking-widest border-4 border-black rounded-none ${getStatusStyles(item.status)}`}
                        >
                          {item.status === "completed"
                            ? "COMPLETE"
                            : item.status === "in-progress"
                            ? "IN PROGRESS"
                            : "PENDING"}
                        </Badge>
                        <span className="text-xs font-black text-black bg-white border-4 border-black px-2 py-1 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transform rotate-2">
                          {item.date}
                        </span>
                      </div>
                      <CardTitle className="text-3xl font-black uppercase tracking-tighter text-black mt-2">
                        {item.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-6 text-sm font-bold text-black leading-snug">
                      <p>{item.content}</p>

                      <div className="mt-6 pt-4 border-t-4 border-black">
                        <div className="flex justify-between items-center text-xs font-black uppercase mb-3 text-black">
                          <span className="flex items-center">
                            <Zap size={16} className="mr-2" strokeWidth={3} />
                            Energy Level
                          </span>
                          <span className="text-lg">{item.energy}%</span>
                        </div>
                        <div className="w-full h-6 bg-white border-4 border-black rounded-none overflow-hidden p-0.5">
                          <div
                            className="h-full bg-black"
                            style={{ width: `${item.energy}%` }}
                          ></div>
                        </div>
                      </div>

                      {item.relatedIds.length > 0 && (
                        <div className="mt-6 pt-4 border-t-4 border-black">
                          <div className="flex items-center mb-4 text-black">
                            <Link size={16} className="mr-2" strokeWidth={3} />
                            <h4 className="text-sm font-black uppercase tracking-widest">
                              Connected Nodes
                            </h4>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {item.relatedIds.map((relatedId) => {
                              const relatedItem = timelineData.find(
                                (i) => i.id === relatedId
                              );
                              return (
                                <Button
                                  key={relatedId}
                                  variant="outline"
                                  className="flex items-center h-10 px-4 py-0 text-xs font-black uppercase rounded-none border-4 border-black bg-white hover:bg-background hover:text-black transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-y-1 hover:translate-x-1 hover:shadow-none"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleItem(relatedId);
                                  }}
                                >
                                  {relatedItem?.title}
                                  <ArrowRight
                                    size={14}
                                    className="ml-2"
                                    strokeWidth={3}
                                  />
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
